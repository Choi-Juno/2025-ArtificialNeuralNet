import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# === 신경망 구조 ===
# 입력층: 3개 뉴런 (x)
# 은닉층: 2개 뉴런 (h11, h12)
# 출력층: 1개 뉴런 (y)

print("=== 첫번째 순전파 시작 ===")
v = np.array([[0.1, 0.2, -0.5], [0.3, 0.2, -0.3]])  # 입력층-은닉층 가중치
x = np.array([0, 0, 1])  # 입력값
print("입력값 (x):", x)
print("입력층-은닉층 가중치 (v):\n", v)

z1 = np.dot(v, x)  # 은닉층 입력값
h11 = sigmoid(z1[0])  # 첫번째 은닉 뉴런 출력
h12 = sigmoid(z1[1])  # 두번째 은닉 뉴런 출력

print("은닉층 출력값 (h11, h12):", h11, h12)

w2 = np.array([0.1, 0.2])  # 은닉층-출력층 가중치
print("은닉층-출력층 가중치 (w2):", w2)
z2 = np.dot(w2, np.array([h11, h12]))  # 출력층 입력값
y = sigmoid(z2)  # 최종 출력
print("출력층 입력값 (z2):", z2)
print("최종 출력값 (y):", y)


def loss(y, t):
    """평균 제곱 오차 (Mean Squared Error)"""
    return 1 / 2 * np.sum(y - t) ** 2


target = 0  # 목표값
print("\n=== 역전파 시작 ===")
print("목표값 (target):", target)
initial_loss = loss(target, y)
print("초기 손실값:", initial_loss)

# 출력층 오차 신호
dy = (target - y) * y * (1 - y)
print("출력층 오차 신호 (dy):", dy)

# 은닉층-출력층 가중치 변화량
dw = 1 * dy * np.array([h11, h12])  # 학습률 1 사용

# 첫번째 입력층과 은닉층 사이의 가중치 변화량 dV 계산
dz1 = np.zeros_like(z1)
dz1[0] = dy * w2[0] * h11 * (1 - h11)
dz1[1] = dy * w2[1] * h12 * (1 - h12)
dV = np.outer(dz1, x)
print("\n=== 역전파 계산 결과 ===")
print("입력층-은닉층 가중치 변화량 (dV):\n", dV)
print("은닉층-출력층 가중치 변화량 (dw):", dw)

# 두번째 순전파를 위해 갱신된 가중치 적용
v_new = v - dV  # 예시로 학습률 1로 사용

# 다시 순전파
z1_new = np.dot(v_new, x)
h11_new = sigmoid(z1_new[0])
h12_new = sigmoid(z1_new[1])

w2_new = w2 - dw  # 예시로 학습률 1로 사용
z2_new = np.dot(w2_new, np.array([h11_new, h12_new]))
y_new = sigmoid(z2_new)

print("\n=== 가중치 업데이트 후 두번째 순전파 ===")
print("업데이트된 출력값 (y_new):", y_new)
print("이전 출력값 (y):", y)

# 오차신호 계산
error_old = loss(target, y)  # 첫번째 순전파 오차
error_new = loss(target, y_new)  # t=0이라 가정
print("\n=== 오차 비교 ===")
print("첫번째 순전파 오차:", error_old)
print("두번째 순전파 오차:", error_new)
print("오차 감소량:", error_old - error_new)
