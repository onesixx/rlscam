import numpy as np
import matplotlib.pyplot as plt

# MATLAB의 rand 함수 대신에 numpy의 random 함수를 사용합니다.
np.random.seed(666)  # 랜덤 시드 설정 (선택 사항)
iter = 1000

# rand 함수 대신에 numpy의 random 함수를 사용하여 점을 생성합니다.
point = np.random.rand(iter, 2)

# 원 안에 있는지 여부를 판단합니다.
circle_in_out = np.sum(point ** 2, axis=1) < 1

# 인덱스를 찾습니다.
circle_in_idx = np.where(circle_in_out == 1)
circle_out_idx = np.where(circle_in_out == 0)

# 원 안에 있는 점의 개수를 계산합니다.
circle_in_count = len(circle_in_idx[0])

# 원주율(Pi)를 계산합니다.
Pi = circle_in_count / iter * 4
# 결과 출력
print(f'=== Iteration : {iter}, Pi = {Pi}, accuracy : {(1 - abs(1 - (Pi / np.pi))) * 100}% ===')

# 원 안에 있는 점과 밖에 있는 점을 시각화합니다.
plt.scatter(point[circle_in_idx,  0], point[circle_in_idx,  1], c='b', marker='.')
plt.scatter(point[circle_out_idx, 0], point[circle_out_idx, 1], c='r', marker='.')
plt.title('Monte-Carlo Pi')
# 그래프 표시
plt.show()


