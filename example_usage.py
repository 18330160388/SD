from k_t_calculator import compute_semantic_curvature

# 示例：外部调用计算K(t)
text = "他打了个电话给朋友，约好晚上一起吃饭。"
results = compute_semantic_curvature(text)

print(f"句子: {text}")
print("结果:")
for r in results:
    print(f"  {r['token']}: K_norm={r['k_norm_t']:.3f} ({r['interpretation']})")