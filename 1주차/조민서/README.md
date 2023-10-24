# hand_math
MNIST dataset활용해서 학습한 손글씨 숫자 인식 toy project

1. ui.py 실행

2. 나온 덧셈 문제를 풀어 paper window에 쓰기

3. submit 버튼 누르기

4. 정답 확인

5. Reset버튼 눌러 새 문제 받기, 반복

![Alt text](./assets/example_img.png)

model: Efficientnet b0 절반만 떼서 사용. Colab에서 학습

Loss fn: CrossEntropyLoss

Optimizer: Adam

Epoch: 30

for more train detail: model.ipynb
