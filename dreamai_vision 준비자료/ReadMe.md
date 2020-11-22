# [꿈꾸는아이](https://dreamai.kr/fair_intel) 경진대회 준비자료
GIST 주최의 꿈꾸는아이 경진대회를 준비하며 시도한 내용, 공부한 내용을 적었다.

## 학습 방법
드론에 적용할 Image Classification Model을 설계하고 Transfer Learning을 통해 학습시켰다.  
1. 학습 이미지  
class별로 train 90장, valid 30장  
(class: 'airplane', 'bird', 'boat', 'bottle', 'car','cat', 'dog', 'drone', 'hamburger', 'horse', 'logo', 'minions', 'paper', 'person', 'rock', 'scissors', 'train'의 17가지)  

2. Network: NASNetMobile  
드론 이미지에서 0.9 이상의 confidence로 물체를 감지할 수 있어야 한다. 따라서, 정확성보다는 loss를 비교하였다.  
다양하게 시도해 본 결과, 제공된 MobileNet에 비해 NASNetMobile이 정확도, loss 모두 뛰어나서 NASNetMobile을 사용하였다.   
(NASNetMobile은 MobileNet에 비해 무겁지만, 드론을 작동시키는데 문제가 없어 이를 사용하였다. 다른 모델을 드론이 작동하기까지 시간이 너무 오래 걸렸다.)  

3. ImageDataGenerator를 사용하여 데이터 전처리, Augmentation  
 처음에는 드론에서의 환경을 고려하여 shift_range를 0.5로 설정하였다. 그런데 물체가 화면 위에 있지 않는 문제가 발생하여 최종 accuracy, loss가 너무 낮아졌다.  
 그래서 rotation_range, shift_range, shift_range, shear_range, zoom_range, brightness_range, horizontal_flip, Brightness_range를 0.1~0.2만큼 설정하여 학습시켰고, 
 물체를 좀 더 잘 인식함을 확인하였다.  

4. Transfer Learning  
마지막 3개의 layer를 제외하고 모든 layer를 동결시켰다. 아주 작은 learning rate(0.0001)로 학습  
overfitting 직전까지 학습하였다. (training은 줄지만 validation은 error가 증가하기 시작하는 지점)  

## 개선 방법
1. 처음부터 Model 짜기  
NASNetMobile Network를 불러올 때, weights='imagenet'으로 하여 불러왔다. 이 때문에, imagenet의 그림이 아니었던 묵찌빠, logo 등의 데이터는 확실하게 인지하지 못하는 문제가 생겼다.
Transfer Learning 대신 초기 weight가 없는 Model을 사용한다면 

2. 학습 방법 변경  
더 강한 이미지 augmentation이나 정규화, CV Layer 조정을 통해 성능을 더 향상시킬 수 있을 것이다.

3. Train, Valid Data 수 늘리기, 확실하게 분류  
특히, 다양한 묵찌빠 Data를 얻기가 어려웠다. 길거리, 건물 안 등 여러 상황에서 묵찌빠 사진을 촬영하여 보충한다면, 더 효과적인 image classificator가 될 것이다.
Test set과 Train set은 구분이 잘 되어야 함
Validation Set은 Test set을 잘 대표할 수 있어야 한다.

## 그 외 시도해 본 내용
코드는 작성하였지만, 사용하지 않았던 학습 방법들이다.
[bag of tricks for image classification with convolutional neural networks](https://openaccess.thecvf.com/content_CVPR_2019/papers/He_Bag_of_Tricks_for_Image_Classification_with_Convolutional_Neural_Networks_CVPR_2019_paper.pdf)
를 참고하였다.  
- Learning rate warm up  
앞의 5 epoch 동안은 LR을 급격하게 증가시키고 Cos 꼴로 점점 줄이기
- CutMix  
드론 영상의 특성상, 이미지 일부 지역이 잘리게 되는데 이를 방지하기 위해 사용하고자 하였다. CutMix를 사용하면 특징적인 부분만 보고도 명확하게 물체 인식을 할 수 있어 loaclization 성능이 향상된다.

## 코드 내용 설명
1. DataGenerator_preview  
DataGenerator를 거쳤을 때 이미지가 어떻게 변화하는지 확인하는 코드

2. mobilenetv2(원본)  
Mobilenetv2를 transfer learning하는 코드

3. mobilenetv2_cutmix  
cutmix를 적용하여 mobilenetv2를 학습한 코드 (channel이 RGB로 3개이고 파일형식이 jpg인 이미지만 사용 가능)

4. NASNetMobile  
NASNetMobile을 transfer learning하는 코드

5. train, valid split  
images 파일에 class 별로 분류되어 있는 이미지를 train, valid dataset으로 분류하는 코드

6. VOC2012 data preprocessing  
[VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) 이미지를 xml file의 class에 따라 분류하여 복사, 붙여넣기하는 코드
