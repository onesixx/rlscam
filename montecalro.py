import random

def estimate_pi(num_samples):
    inside_circle = 0
    
    for _ in range(num_samples):
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        
        # 원의 중심으로부터의 거리를 계산
        distance = x**2 + y**2
        
        # 원 안에 있는 경우
        if distance <= 1:
            inside_circle += 1
    
    # 원의 면적을 추정
    pi_estimate = (inside_circle / num_samples) * 4
    return pi_estimate

# 시뮬레이션에 사용할 샘플 수를 정의
num_samples = 1000000

# 원주율 값을 추정
pi_estimate = estimate_pi(num_samples)

print(f"Monte Carlo 시뮬레이션을 통한 원주율 추정: {pi_estimate}")
