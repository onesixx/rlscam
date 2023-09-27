import torch
import torch.nn as nn
import torch.optim as optim

# 신경망 모델 정의
class FeedForwardNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # 첫 번째 fully connected layer
        self.relu = nn.ReLU()  # 활성화 함수 (ReLU)
        self.fc2 = nn.Linear(hidden_size, output_size)  # 두 번째 fully connected layer

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 모델 초기화
input_size = 3  # 입력 특성의 개수
hidden_size = 4  # 은닉층의 뉴런 개수
output_size = 2  # 출력 클래스 개수
model = FeedForwardNN(input_size, hidden_size, output_size)

# 손실 함수와 옵티마이저 정의
criterion = nn.CrossEntropyLoss()  # 분류 작업을 위한 크로스 엔트로피 손실
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 확률적 경사 하강법을 사용하는 옵티마이저

# 훈련 데이터와 레이블 생성
# 여기에서는 임의의 데이터와 레이블을 사용하는 가상의 예제입니다.
# 실제 데이터에 대해서는 적절한 데이터를 로드하고 전처리해야 합니다.
x_train = torch.randn(100, input_size)
y_train = torch.randint(0, output_size, (100,))

# 모델 훈련
num_epochs = 100
for epoch in range(num_epochs):
    # 순전파
    outputs = model(x_train)
    
    # 손실 계산
    loss = criterion(outputs, y_train)
    
    # 역전파 및 가중치 업데이트
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 매 에폭마다 손실 출력
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 모델 평가
# 평가 데이터와 평가 과정을 추가해야 합니다.
# 이 예제에서는 생략했습니다.
