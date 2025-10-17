# app.py  —  亲属相似度对比（Streamlit）
# 模型栈：InsightFace (ArcFace + SCRFD, ONNXRuntime)  —— 轻量、云端稳定
# 特色：支持 FIW 亲属分类头（joblib），无权重时回退启发式百分比

import os
import json
import numpy as np
import streamlit as st
from PIL import Image
from insightface.app import FaceAnalysis

# ======== 基础度量 & 启发式百分比 ========
def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))

def l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

def kinship_percent_heuristic(cos_sim: float, l2d: float) -> dict:
    # 与我们之前 CLI 版一致的启发式：先将 Cosine/L2 归一化，再融合成 score，最后分段分配概率
    cos_cal = float(np.clip((cos_sim - 0.20) / (0.98 - 0.20), 0, 1))
    l2_cal  = float(np.clip((0.90 - l2d) / (0.90 - 0.20), 0, 1))
    score   = 0.6 * cos_cal + 0.4 * l2_cal

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
        "cos": float(cos_sim),
        "l2": float(l2d),
        "pc": float(v[0]),     # parent_child
        "sb": float(v[1]),     # siblings
        "nk": float(v[2]),     # non-kin
    }

# ======== FIW 分类头（可选） ========
try:
    import joblib
    from sklearn.preprocessing import StandardScaler
    FIW_AVAILABLE = True
except Exception:
    FIW_AVAILABLE = False

def build_pair_features(e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
    # 特征：e1, e2, |e1-e2|, e1*e2, 以及全局指标 cos/l2
    e1 = e1.astype(np.float32)
    e2 = e2.astype(np.float32)
    diff = np.abs(e1 - e2)
    prod = e1 * e2
    c = cosine(e1, e2)
    d = l2(e1, e2)
    feat = np.concatenate([e1, e2, diff, prod, np.array([c, d], dtype=np.float32)], axis=0)
    return feat

@st.cache_resource
def load_fiw_head():
    """从 ./models/fiw_head.joblib / fiw_scaler.joblib 加载分类器与标准化器。"""
    if not FIW_AVAILABLE:
        return None, None
    base = os.path.dirname(__file__)
    model_path  = os.path.join(base, "models", "fiw_head.joblib")
    scaler_path = os.path.join(base, "models", "fiw_scaler.joblib")
    if not os.path.exists(model_path):
        return None, None
    clf = joblib.load(model_path)
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    return clf, scaler

def fiw_predict_proba(e1: np.ndarray, e2: np.ndarray, clf, scaler) -> dict:
    x = build_pair_features(e1, e2).reshape(1, -1)
    if scaler is not None:
        x = scaler.transform(x)
    else:
        # 没有 scaler 就在线标准化一次以稳住数值尺度
        s = StandardScaler()
        x = s.fit_transform(x)

    # 取概率输出（若是 SVM 无概率，则用 decision_function + softmax 近似）
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(x)[0]
        classes = list(getattr(clf, "classes_", []))
    elif hasattr(clf, "decision_function"):
        z = clf.decision_function(x)
        z = z.reshape(-1) if hasattr(z, "shape") else np.array(z, dtype=np.float32)
        exp = np.exp(z - np.max(z))
        proba = exp / np.sum(exp)
        classes = list(range(len(proba)))
    else:
        raise RuntimeError("Classifier does not support probability outputs.")

    # 统一映射到 pc/sb/nk
    idx_map = {}
    for i, c in enumerate(classes):
        key = str(c).lower()
        if key in ["parent_child", "pc", "0"]:
            idx_map["pc"] = i
        elif key in ["siblings", "sb", "1"]:
            idx_map["sb"] = i
        elif key in ["nonkin", "nk", "2", "non_kin", "non-kin"]:
            idx_map["nk"] = i
    pc = float(proba[idx_map.get("pc", 0)])
    sb = float(proba[idx_map.get("sb", 1 if len(proba) > 1 else 0)])
    nk = float(proba[idx_map.get("nk", 2 if len(proba) > 2 else -1)]) if len(proba) > 2 else float(1.0 - pc - sb)
    tot = max(pc + sb + nk, 1e-9)
    return {"pc": pc / tot, "sb": sb / tot, "nk": nk / tot}

# ======== InsightFace 模型（检测 + 特征） ========
@st.cache_resource
def load_insightface():
    # 官方 “buffalo_l”：SCRFD 检测 + ArcFace(ResNet100) 特征；CPUExecutionProvider 默认可用
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app

def get_embedding_from_image(img: Image.Image, app: FaceAnalysis) -> np.ndarray:
    # InsightFace 期望 BGR ndarray；PIL 是 RGB，需要转换
    arr = np.array(img.convert("RGB"))[:, :, ::-1]  # RGB -> BGR
    faces = app.get(arr)
    if not faces:
        raise RuntimeError("未检测到人脸，请上传更清晰的正脸照片。")
    # 取最大的人脸
    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
    # insightface 已返回归一化后的 embedding（normed_embedding）
    emb = getattr(face, "normed_embedding", None)
    if emb is None:
        emb = face.embedding
        emb = emb.astype(np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-9)
    else:
        emb = emb.astype(np.float32)
    return emb

# ======== Streamlit UI ========
st.set_page_config(page_title="👨‍👩‍👧 亲属相似度（FIW可选）", layout="centered")
st.title("👨‍👩‍👧 亲属相似度对比（InsightFace + 可选 FIW 分类头）")
st.caption("建议使用清晰、正脸、无遮挡的照片。Developed by c8geek")

app = load_insightface()
c1, c2 = st.columns(2)
f1 = c1.file_uploader("上传照片 A", type=["jpg", "jpeg", "png"])
f2 = c2.file_uploader("上传照片 B", type=["jpg", "jpeg", "png"])

use_fiw_default = True if FIW_AVAILABLE else False
use_fiw = st.checkbox("使用 FIW 亲属分类器（需在 main/models/ 放 fiw_head.joblib）", value=use_fiw_default)
clf, scaler = load_fiw_head() if use_fiw else (None, None)

if f1 and f2:
    img1 = Image.open(f1)
    img2 = Image.open(f2)
    st.image([img1, img2], caption=["照片 A", "照片 B"], use_column_width=True)

    with st.spinner("正在提取特征向量…"):
        emb1 = get_embedding_from_image(img1, app)
        emb2 = get_embedding_from_image(img2, app)
        cs = cosine(emb1, emb2)
        l2d = l2(emb1, emb2)

        if clf is not None:
            # 使用 FIW 分类头：概率来自已训练模型，score 仅作参考展示
            cos_cal = float(np.clip((cs - 0.20) / (0.98 - 0.20), 0, 1))
            l2_cal  = float(np.clip((0.90 - l2d) / (0.90 - 0.20), 0, 1))
            score   = 0.6 * cos_cal + 0.4 * l2_cal
            proba   = fiw_predict_proba(emb1, emb2, clf, scaler)
            out = {"cos": cs, "l2": l2d, "score": score, "pc": proba["pc"], "sb": proba["sb"], "nk": proba["nk"]}
            st.info("已启用 FIW 分类头（概率来自数据驱动模型）。")
        else:
            out = kinship_percent_heuristic(cs, l2d)
            st.caption("未检测到 FIW 分类器，使用启发式概率映射。可将模型放在 main/models/fiw_head.joblib。")

    st.subheader("结果")
    st.write(f"**Cosine 相似度**：{out['cos']:.4f}")
    st.write(f"**L2 距离**：{out['l2']:.4f}")
    st.write(f"**综合分数（参考）**：{out['score']:.4f}（≥ 0.72 通常判为高概率亲属）")
    st.progress(min(max((out['score'] - 0.2) / 0.78, 0), 1))

    st.markdown("**亲属概率**")
    st.write(f"• 父母子女：**{out['pc']*100:.1f}%**")
    st.write(f"• 兄弟姐妹：**{out['sb']*100:.1f}%**")
    st.write(f"• 非亲属：**{out['nk']*100:.1f}%**")

    if out["score"] >= 0.72:
        st.success("高概率为亲属（更偏父母子女）。建议用不同角度/年龄阶段照片交叉验证。")
    elif out["score"] >= 0.60:
        st.warning("存在亲属可能，但证据一般；建议使用更清晰、角度更接近的照片复核。")
    else:
        st.info("大概率非亲属；若年龄差很大也可能分数偏低。")

    # 导出 JSON
    result = {
        "cosine": out["cos"],
        "l2": out["l2"],
        "score": out["score"],
        "probabilities": {"parent_child": out["pc"], "siblings": out["sb"], "nonkin": out["nk"]},
    }
    st.download_button(
        "下载结果 JSON",
        data=json.dumps(result, ensure_ascii=False, indent=2),
        file_name="kinship_result.json",
        mime="application/json",
    )
else:
    st.info("请上传两张照片进行对比。")
