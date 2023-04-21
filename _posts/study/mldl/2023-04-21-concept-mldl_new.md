---
layout: post
bigtitle:  "면접을 위한 내 전공 공부1: 기계학습"
subtitle:   "용어 정리"
categories:
    - study
    - mldl
tags:
    - ML
comments: true
published: true
---

직무면접을 위해 배운 전공수업 내용을 정리하거나 졸업시험 준비할 때 정리해둔 내용을 작성해서 면접때 까지 볼까 한다. 

### 딥러닝이 무엇인가요? 머신러닝 알고리즘과 비교하여 어떻게 다른가요? 
딥러닝은 머신러닝의 서브셋이며 backpropagation과 large set을 정확하게 모델링하는 특정 원칙을 어떻게 사용할지 고려하는 neural network입니다. <br>
또한 머신러닝은 엔지니어가 개입하여 조정해야하는 반면 딥러닝은 알고리즘 자체 신경망을 통해 예측 정확성 여부를 스스로 판단합니다.                <br>

예 : 개와 고양이 분류 
(딥러닝- 이미지에서 스스로 feature를 찾고 그것을 학습하여 모델링, 머신러닝은 엔지니어가 개와 고양이를 분류할만한 기준을 정해 데이터를 나누고 (꼬리의 길이, 귀의 모양등) 그 특징 데이터를 학습시킵니다.)



### Supervised와 Unsupervised machine learning의 차이는 무엇입니까? 

supervised learning은 training labeled data가 필요합니다.

unsupervised learning은 명시적인 labeled data를 요구하지 않습니다.



### Roc curve가 무엇이고 어떻게 작동하는지 설명하세요 

Roc curve는 이진분류기의 성능을 그래픽하게 표현한 커브로 가능한 모든 threshold에 대한 true positive rate(Y축)와 false positive rate(X축)의 비율을 표현한것입니다.
두 클래스를 더 잘 구별할 수 있다면 Roc curve는 좌상단에 더 가까워집니다. 

### Precision과 Recall을 정의하세요 

Precision은 모델이 True라고 분류한것중 실제 True인 것의 비율로 모델 중심적 평가이며  <br>
Recall은 실제 True인것중에 모델이 True라고 예측한것의 비율로 데이터 중심적 평가입니다. <br>
즉 Precision은 true라고 예측한것중 실제 true의 비율, recall은 실제 true중에 true라고 예측한것의 비율입니다. 

### Precision과 Recall을 통해 무엇을 이해할수 있습니까?

모델중심의 평가와 데이터중심의 평가를 통해 모델을 더 다양한 방면으로 평가하고 이해할수 있습니다.


### False positive는 무엇입니까?, False negative는 무엇입니까?

False positive: 모델이 positive라고 예측했는데 실제 데이터의 정답은 False인 경우입니다.

False negative: 모델이 negative라고 예측했는데 실제 데이터의 정답은 True인 경우입니다.

### Bayes’ Theorem이 무엇입니까? 또한 machine learning분야에서 어떻게 유용합니까?

베이즈 이론은 prior knowledge를 알고 있는 주어진 event의 posterior probability를 제공합니다.            <br>
Machine learning분야에서는 데이터가 주어졌을 때 레이블이 나타날 확률을 구하는데 사용될 수 있습니다.         <br>
또는 데이터가 주어졌을 때 파라메타를 구할 수 있게 되어 데이터 기반 학습을 가능하게 해줍니다.                <br>


예를들어, 암이 나이와 관련이 있다고 가정하면, bayes’ theorem을 이용하면 사람의 나이를 모르는 상태에서 암을 예측하는것보다 나이를 알고 있을때 암에 걸릴 확률을 더 정확하게 예측할 수 있습니다. 


### 왜 Naive Bayes는 naive한가요?

Naive bayes는 input이 주어졌을때 target class를 output합니다.  <br>
bayes’ theorem에서 분모를 무시하고 분자만 계산하는것으로 분모(prior)는 클래스 정보를 가지고 있지 않아 상수로 취급할 수 있기 때문입니다. <br>
또한 feature들 간의 의존관계를 무시하고 feature들이 서로 완전히 독립적이라는 나이브한 가정을 베이즈 정리에 적용하기 때문에 나이브베이즈라고 합니다. <br>

### F1 score는 무엇입니까? 그리고 어떻게 사용할 수 있나요?

F1 score는 model의 performance의 측정입니다. 이 값은 model의 precision과 recall의 weighted average(가중치평균)으로 결과는 1이 best, 0이 worst입니다.
classification class간의 데이터 분균형이 심할 때 사용합니다. 

### regression 대신 언제 classification을 사용할 수 있습니까?

dataset에서 label이 있는 catagorical data로 나누어져있을때, classification을 사용할 수 있습니다. 


### Model이 overfitting이 되지 않았음을 어떻게 확신하는가? 또 어떻게 해결할 수 있는가?

먼저 overfitting이란 Training data set에 대해 과하게 학습된 상황이며 training data set 이외의 데이터에 대해선 모델이 잘 동작하지 못하는 것을 말합니다. 
즉 training data set에 대해서 정답을 완벽하게 외워서 generalization을 못하는 경우를 말합니다.

overfitting을 방지하기 위한 방법으로 
1.	variance를 줄임으로써 모델을 단순화 합니다.
2.	cross-validation 기술을 사용합니다.
3.	model의 특정 parameter가 overfitting을 야기하는것 같다면 model의 특정 parameter에 패널티를 가하는 regularization을 사용합니다.
4.	학습데이터를 늘립니다.
5.	Dropout을 사용합니다.
6.	Early stopping을 사용합니다.


### Machine learning model의 effectiveness를 판단하기 위해 어떤 evaluation approaches를 사용해야 합니까? 

보통 dataset을 training set과 test set으로 나누는 이유는 model의 effectiveness를 평가하기 위해서 입니다.
이때 model의 effectiveness를 평가하는 방법으로 accuracy, precision, recall, F1 score와 같은 측정 방법을 통해 model을 적절한 상황에 적합한 성능을 측정할 수 있는 방법을 사용해야 합니다

### Logistic regression model을 어떻게 평가할 수 있나요? 

Logistic regression model은 선형알고리즘에 sigmoid function이 결합된 분류알고리즘입니다.          <br>
선형 회귀방식을 이용한 이진분류 알고리즘이므로 분류 레이블은 0과 1입니다.                             <br>
이때, 평가방법은 confusion matrix, precision, recall, F1 score를 이용할 수 있습니다.              <br>


### Regression에서 Error term을 구성하는것은 무엇인가요?

Error는 bias error, variance error, irreducible error로 구성되어 있습니다.

bias error와 variance error는 줄일 수 있지만 irreducible error는 줄일 수 없습니다.


### time-series data를 작업할 때, 어떤 sampling 기술이 가장 적합한가요?

시계열 데이터로 작업할 때, 가장 적합한 샘플링 기술은 지속적으로 추가되는 custom iterative sampling입니다. 
검증(validation)에 사용된 sample이 다음 train sets에 추가되고 새로운 sample이 검증(validation)을 위해 사용되는 형태로 할 수 있습니다. 

### Kernel trick이 무엇인가요? 그리고 어떻게 유용한가요?

저차원 데이터에 대해 svm을 해결하지 못하는 경우 저차원을 고차원으로 매핑함으로써 문제를 해결할 수 있는데 이때 데이터를 저차원에서 고차원으로 매핑해주는 함수를 kernel function이라고 합니다. 
SVM에 kernel function을 도입하게 되면 모든 관측치에 대해 고차원으로 매핑하고 내적해야하기 때문에 연산량이 폭증합니다. 따라서 고차원 매핑과 내적을 한번에 해결하기위해 도입된 것이 kernel trick입니다. 
즉, 저차원공간에서 고차원으로 매핑하여 행렬 내적을 한번에 하는 방법입니다.

### L1과 L2 regularization을 통해 무엇을 이해할수 있습니까?

regularization의 목적은 매끄럽지않은 데이터 설명 함수를 매끄럽게 만드는것이며 모델의 overfitting을 막는 방법입니다.
L1 regularization은 자잘한 가중치 파라메타는 0으로 만들고 중요한 가중치만 남아 파라메타의 수를 줄여 overfitting을 막습니다.
L2 regularization은 큰 가중치에 대해서는 규제를 강하게 하고 작은 가중치에 대해서는 규제를 약하게 주어 모든 가중치들이 모델에 고르게 반영되도록합니다. 따라서 가중치를 0으로 수렴하는 비율이 L1에 비해 작습니다.

### Normal distribution이 무엇인가요?

다음과 같은 특징을 갖는 distribution을 normal distribution이라고 부릅니다.
-	mean, mode, median이 모두 같다
-	curve가 중심을 기준으로 대칭이다
-	정확하게 중심을 기준으로 왼쪽 절반과 오른쪽 절반이 같다
-	curve아래의 넓이의 합이 1이다.

### Classification problem에서 class imbalance를 어떻게 다루어야하나요?

-	Class weights를 사용합니다
-	Sampling을 사용합니다
-	SMOTE를 사용합니다 
-	Focal Loss와 같은 loss functions를 선택합니다 

이때 SMOTE는 대표적인 오버샘플링 기법중 하나로 전체 데이터중 낮은 비율로 존재하는 클래스의 데이터를 K-NN 알고리즘을 이용해서 데이터를 생성하는 방법입니다.

Focal loss는 분균형한 클래스문제를 해결하기 위해 최종 확률값이 낮은 hard class에서 loss를 조금만 줄이는 역할을 합니다. 또 오분류되는 케이스에 대해서는 더 큰 가중치를 주는 방법입니다. 좀더 문제가 있는 loss에 더 집중합니다.


### Generative model과 Discriminative model의 차이점을 설명해주세요.

generative model은 데이터 $X$가 생성되는 과정을 두개의 확률모형 $(P(Y),P(X|Y))$을 정의하고 
Bayes’ Rule을 사용해 $P(Y|X)$를 간접적으로 도출하는 모델을 가리킵니다. 

generative model은 레이블이 있는경우 지도학습기반의 generative model이라고 하며 선형판별분석이 대표적입니다. 
레이블이 없는경우 비지도학습기반의 generative model이라고 하며 가우시안믹스쳐모델이 대표적입니다.

Discriminative model은 데이터 $X$가 주어졌을때 레이블 $Y$가 나타날 조건부확률 $P(Y|X)$를 직접적으로 반환하는 모델을 가리킵니다. 레이블 정보가 있어야하기 때문에 지도학습범주에 속하며 $X$의 레이블을 잘 구분하는 결정경계를 학습하는것이 목표가 됩니다. 선형회귀와 로지스틱회귀가 대표적입니다.


### Hyperparameters는 무엇이며 parameters와 어떻게 다른가요?
학습된 모델의 일부분으로 저장되는 weights, bias는 parameter로써 모델의 내부에 있고,
hyperparameter는 데이터에서 값을 추정할 수 있는 모델의 외부에 있는 변수로 종종 최적의 parameter를 추정하는데 사용됩니다. learning rate등이 여기에 속합니다. 

### Inductive Bias의 개념을 정의하고 설명해보세요.
Inductive Bias는 learning algorithm이 아직 경험하지 못한 입력이 주어졌을 때, 출력을 예측하기 위해 사용하는 가정입니다. 
X로부터 Y를 배우려고 할 때, Y에 대한 가설공간이 무한하다면, Inductive bias라 불리는 가설공간에 대한 가정을 통해 그 범위를 줄일 필요가 있습니다.
예를 들어 CNN 모델의 경우 input으로 들어온 데이터가 이미지일 것이고 이미지에 대한 예측 모델일 것이라는 가정과 하나의 픽셀이 그 픽셀의 주변과 관련 있을 것이라는 가설을 설정합니다.
RNN의 경우도 마찬가지로 input으로 들어오는 데이터가 문장일 것이라고 가정하고 관련 단어가 다음단계에 영향을 줄 것이라고 가정합니다.
반면 Transformer의 경우 비전과 nlp분야등 다양하게 사용되며 모든 공간정보 또는 위치정보를 사용합니다.

따라서 transformer는 inductive bias가 낮으며 robust하다고 합니다.

### 가장 보편적인 차원축소(dimensionality reduction) 알고리즘을 말하고 설명해보세요.

PCA는 분포된 데이터들의 주성분을 찾아주는 방법입니다. 즉 분포의 주성분을 분석하는 방법입니다.       <br>
이때 주성분의 뜻은 데이터들의 분산이 가장 큰 방향벡터를 의미합니다.       <br>
PCA의 목적은 다차원에서 데이터의 차원을 감소시키는것이 주 목적입니다.       <br>
PCA의 목적은 원래 데이터의 분산을 최대한 보존하는 축을 찾아 투영해 차원을 줄이는것입니다.        <br>
데이터에 가장 가까운 초평면을 구한 다음, 데이터를 이 초평면에 투영시킵니다.       <br>
초평면에 데이터를 투영하기전에 먼저 적절한 초평면을 선택해야합니다. 데이터분산이 최대가 되는 축, 즉 원본 데이터셋과 투영된 데이터셋의 평균제곱거리를 최소화하는 축을 찾습니다.       <br>
1. 학습데이터셋에서 분산이 최대인 축을 찾는다
2. 이렇게 찾은 첫번째 축과 직교하면서 분산이 최대인 두번째 축을 찾는다
3. 첫번째 축과 두번째 축과 직교하면서 분산을 최대한 보존하는 세번째 축을 찾는다.
4. 1~3과 같은 방법으로 데이터셋의 차원만큼의 축을 찾는다 
