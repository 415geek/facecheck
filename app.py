import numpy as np, streamlit as st
from PIL import Image
from insightface.app import FaceAnalysis

def cosine(a,b): return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))
def l2(a,b): return float(np.linalg.norm(a-b))

def kinship_percent(cos_sim,l2d):
    cos_cal = float(np.clip((cos_sim-0.20)/(0.98-0.20),0,1))
    l2_cal  = float(np.clip((0.90-l2d)/(0.90-0.20),0,1))
    score = 0.6*cos_cal + 0.4*l2_cal
    if score>=0.72:
        pc=0.65*score+0.25; sb=0.30*score+0.15; nk=1-(pc+sb)
    elif score>=0.60:
        pc=0.45*score+0.15; sb=0.40*score+0.10; nk=1-(pc+sb)
    else:
        nk=0.80-0.6*score; pc=0.12+0.2*score; sb=1-(nk+pc)
    v=np.clip(np.array([pc,sb,nk],dtype=np.float32),1e-6,1.0); v/=v.sum()
    return dict(score=float(score), cos=float(cos_sim), l2=float(l2d),
                pc=float(v[0]), sb=float(v[1]), nk=float(v[2]))

@st.cache_resource
def load_app():
    # CPU 上的轻量模型；insightface 会自动下载模型到 ~/.insightface
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640,640))
    return app

st.set_page_config(page_title="亲属相似度（轻量版）", layout="centered")
st.title("👨‍👩‍👧 亲属相似度（ONNX 轻量版，云端稳定）")
st.caption("不依赖 TensorFlow，部署更稳。建议上传清晰正脸照。")

app = load_app()
c1,c2 = st.columns(2)
f1 = c1.file_uploader("上传照片 A", type=["jpg","jpeg","png"])
f2 = c2.file_uploader("上传照片 B", type=["jpg","jpeg","png"])

def get_emb(img: Image.Image):
    arr = np.array(img.convert("RGB"))
    faces = app.get(arr)
    if not faces:
        raise RuntimeError("未检测到人脸，请换更清晰/正脸照片")
    # 取最大的脸
    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
    # insightface 返回的是已归一化的 embedding
    return face.normed_embedding.astype(np.float32)

if f1 and f2:
    img1, img2 = Image.open(f1), Image.open(f2)
    st.image([img1, img2], caption=["照片 A","照片 B"], use_column_width=True)
    with st.spinner("计算中…"):
        emb1 = get_emb(img1); emb2 = get_emb(img2)
        cs = cosine(emb1, emb2); l2d = l2(emb1, emb2)
        out = kinship_percent(cs, l2d)
    st.subheader("结果")
    st.write(f"**Cosine 相似度**：{out['cos']:.4f}")
    st.write(f"**L2 距离**：{out['l2']:.4f}")
    st.write(f"**综合分数**：{out['score']:.4f}（≥0.72 高概率亲属）")
    st.progress(min(max((out['score']-0.2)/0.78,0),1))
    st.markdown("**亲属概率估计（启发式）**")
    st.write(f"• 父母子女：**{out['pc']*100:.1f}%**")
    st.write(f"• 兄弟姐妹：**{out['sb']*100:.1f}%**")
    st.write(f"• 非亲属：**{out['nk']*100:.1f}%**")
    if out["score"]>=0.72: st.success("高概率为亲属（更偏父母子女）。")
    elif out["score"]>=0.60: st.warning("存在亲属可能，但证据一般。")
    else: st.info("大概率非亲属。")
else:
    st.info("请上传两张照片进行对比。")
