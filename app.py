import io, os, numpy as np
import streamlit as st
from PIL import Image
from deepface import DeepFace

# ---------- 亲属百分比映射（与你本地脚本一致） ----------
def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def l2_distance(a, b):
    return float(np.linalg.norm(a - b))

def kinship_percentages(cos_sim: float, l2: float):
    cos_cal = (cos_sim - 0.20) / (0.98 - 0.20)
    cos_cal = float(np.clip(cos_cal, 0.0, 1.0))
    l2_cal = (0.90 - l2) / (0.90 - 0.20)
    l2_cal = float(np.clip(l2_cal, 0.0, 1.0))
    score = 0.6 * cos_cal + 0.4 * l2_cal

    if score >= 0.72:
        pc = 0.65 * score + 0.25
        sb = 0.30 * score + 0.15
        nk = 1.0 - (pc + sb)
    elif score >= 0.60:
        pc = 0.45 * score + 0.15
        sb = 0.40 * score + 0.10
        nk = 1.0 - (pc + sb)
    else:
        nk = 0.80 - 0.6 * score
        pc = 0.12 + 0.2 * score
        sb = 1.0 - (nk + pc)

    v = np.array([pc, sb, nk], dtype=np.float32)
    v = np.clip(v, 1e-6, 1.0)
    v = v / v.sum()
    return {
        "score": float(score),
        "cos_sim": float(cos_sim),
        "l2": float(l2),
        "parent_child": float(v[0]),
        "siblings": float(v[1]),
        "nonkin": float(v[2]),
    }

# ---------- DeepFace 模型缓存 ----------
@st.cache_resource
def load_model():
    DeepFace.build_model("Facenet512")  # 先用 Facenet512（兼容性最好）
    return True

def get_embedding_from_bytes(file_bytes):
    # 将上传文件落到临时文件再让 deepface 读取
    tmp_path = "._tmp_upload.jpg"
    with open(tmp_path, "wb") as f:
        f.write(file_bytes)
    rep = DeepFace.represent(
        img_path=tmp_path,
        model_name="Facenet512",
        detector_backend="opencv",  # 避开 retinaface 对 tf-keras 的要求
        enforce_detection=True
    )
    vec = np.array(rep[0]["embedding"], dtype=np.float32)
    return vec / (np.linalg.norm(vec) + 1e-9)

# ---------- UI ----------
st.set_page_config(page_title="本地亲属相似度（离线）", layout="centered")
st.title("👨‍👩‍👧 本地亲属相似度（离线）")
st.caption("所有计算均在本机进行，不上传网络。建议使用清晰、正脸、无遮挡的照片。")

load_model()

col1, col2 = st.columns(2)
with col1:
    f1 = st.file_uploader("上传照片 A", type=["jpg","jpeg","png"])
with col2:
    f2 = st.file_uploader("上传照片 B", type=["jpg","jpeg","png"])

if f1 and f2:
    img1 = Image.open(f1).convert("RGB")
    img2 = Image.open(f2).convert("RGB")
    st.image([img1, img2], caption=["照片 A","照片 B"], use_column_width=True)

    with st.spinner("正在提取特征…"):
        emb1 = get_embedding_from_bytes(f1.getvalue())
        emb2 = get_embedding_from_bytes(f2.getvalue())
        cos = cosine_similarity(emb1, emb2)
        l2  = l2_distance(emb1, emb2)
        out = kinship_percentages(cos, l2)

    st.subheader("结果")
    st.write(f"**Cosine 相似度**：{out['cos_sim']:.4f}")
    st.write(f"**L2 距离**：{out['l2']:.4f}")
    st.write(f"**综合分数 Score**：{out['score']:.4f}（≥ 0.72 通常判为高概率亲属）")

    prog = min(max((out['score'] - 0.2) / 0.78, 0), 1)
    st.progress(prog)

    pc = out["parent_child"] * 100
    sb = out["siblings"] * 100
    nk = out["nonkin"] * 100

    st.markdown("**亲属概率估计**（启发式）")
    st.write(f"• 父母子女：**{pc:.1f}%**")
    st.write(f"• 兄弟姐妹：**{sb:.1f}%**")
    st.write(f"• 非亲属：**{nk:.1f}%**")

    if out["score"] >= 0.72:
        st.success("高概率为亲属（更偏父母子女）。建议用不同角度/年龄阶段照片交叉验证。")
    elif out["score"] >= 0.60:
        st.warning("存在亲属可能，但证据一般；建议换更清晰/角度更接近的照片复核。")
    else:
        st.info("大概率非亲属；若为远亲或年龄差很大也可能分数偏低。")

    # 导出 JSON
    import json
    result = {
        "cosine": out["cos_sim"],
        "l2": out["l2"],
        "score": out["score"],
        "probabilities": {
            "parent_child": out["parent_child"],
            "siblings": out["siblings"],
            "nonkin": out["nonkin"]
        }
    }
    st.download_button(
        "下载结果 JSON",
        data=json.dumps(result, ensure_ascii=False, indent=2),
        file_name="kinship_result.json",
        mime="application/json"
    )
else:
    st.info("请上传两张照片进行对比。")
