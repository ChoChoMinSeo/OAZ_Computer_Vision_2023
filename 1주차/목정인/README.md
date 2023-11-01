# ResNet 이용한 고양이와 개 분류

학습률 : 0.1, 미니배치: 32, epoch: 100으로 설정 <br>
회전, 아핀변환, color jitter를 통해 Data Augmentation -> train dataset 정규화
input channels: 3, output channels: 64, kernel_size=3, stride=2, padding=3
loss function: CrossEntropyLoss 사용

<img width="550" alt="image" src="https://github.com/koreaoaz/OAZ_Computer_Vision_2023/assets/108310627/1061be18-8e24-459a-a36e-e7abe06b74fa">
결과: test accuracy가 83.9%로 꽤 좋은 성능을 보입니다.



** 전공 수업에서 배운 코드를 참조해서 작성했습니다 
