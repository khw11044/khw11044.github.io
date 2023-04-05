---
layout: post
bigtitle:  "Tacotron: Towards End-to-End Speech Synthesis 요약,정리"
subtitle:   "Tacotron, text-to-speech"
categories:
    - blog
    - papers
    - paper-etc
tags:
    - Tacotron
    - TextToSpeech
comments: true
published: true
related_posts:
    - _posts/blog/githubpage/2021-02-01-tacotron1_expain.md
    - _posts/blog/githubpage/2020-02-02-seq2seq-with-attention.md
    - _posts/blog/githubpage/2020-02-05-tacotron2.md
---

# Tacotron: Towards End-to-End Speech Synthesis 요약, 정리

colab에서 수행 코드

> [colab에서 수행 코드 ](https://github.com/khw11044/Tacotron-in-colab)


## ABSTRACT(초록)

- TTS시스템 : 전형적으로 text analysis frontend, acoustic model, audio 합성 모듈과 같은 multiple stage의 시스템이다.
- 이러한 것들은 특정 도메인의 전문지식이 요구되고 불안정한 디자인을 선택하였다.
- 이 논문에서 Character로부터 직접 음성을 합성하는 end-to-end TTS 모델인 Tacotron을 제안한다.
- <text, audio> pair다 주어지면, 이 모델은 랜덤 초기화 이후 처음부터 완벽하게 학습될수 있다.
- Sequence-to-sequence 프레임워크가 이 어려운 태스크를 위해 잘 동작하기 위한 몇가지 중요한 테크닉들을 제안한다.
- Tacotron은 미국 영어를 타겟으로 5점 만점의 mean opinion score에서 3.82점에 도달하였다. 자연스러움에서 기존 parametric system을 능가한다.
- 게다가, Tacotron은 프레임 수준에서 음성을 생성하기 때문에, 샘플 수준의 autoregressive 방법들보다 빠르다.


## 1 INTRODUCTION

**기존 방법들의 문제점, 단점**

- 현대 TTS 파이프라인은 복잡함.
- 통계학적 parametric TTS는 ‘다양한 언어학적 특성들을 추출하는 text frontend’, ‘음성 길이를 예측하는 모델’, ‘음성 특성을 예측하는 모델’, ‘복잡한 signal processing 기반의 vocoder’가 있다.
- 이러한 컴포넌트들은 전문적인 지식을 기반으로 하고 디자인하기 매우 어려움
- 개별적으로 학습되기 때문에 각 컴포넌트에서 발생한 에러들이 축적됨
- 그러므로 현대 TTS의 복잡도는 새로운 시스템을 구축할 때 상당한 엔지니어적인 노력이 필요

**사람들이 라벨링한 <text,audio> pair만 가지고 학습될수 있는 end-to-end TTS 시스템의 장점**

- heuristic한 실험들과 연관된 feature engineering을 완화할 수 있음
- 화자, 언어, 감정 등과 같은 다양한 특성들을 더욱 쉽게 조절할 수 있다. 왜냐하면 이러한 특성 조절은 특정한 컴포넌트에서 발생하는 것이 아니라 모델의 시작부터 발생하기 때문.
- 하나의 모델은 각 컴포넌트의 에러가 축적되는 multi-stage 모델에 비해 강력
- 이러한 장점들은 end-to-end 모델이 종종 noisy가 있을 수 있지만 실제 세상에서 찾을 수 있는 풍부하고 표현적인 많은 양의 데이터를 학습할 수 있도록 한다.

**어려움**

- TTS는 매우 압축된 text(source)를 오디오로 압축을 푸는 large-scale inverse 문제가 있음
- 동일한 텍스트라도 발음 또는 발화 스타일이 다르기 때문에 end-to-end 모델에서 특히 어렵다.
- 따라서 주어진 입력에 대한 매우 다양한 범위를 가지는 signal level로 대처해야한다.
- end-to-end speech recognition이나 machine translation과 달리, TTS의 outputs은 연속적이고 일반적으로 output sequences는 input보다 훨씬 더 길다. 이러한 속성은 예측 오류를 빠르게 누적시킨다.


**방안**

- 본 논문에서 attention 기반의 sequence-to-sequence 모델을 사용한 end-to-end TTS 모델, Tacotron을 제안한다.
- 이 모델의 입력은 characters, 출력은 raw spectrogram이고, 몇가지 테크닉들을 사용하여 모델의 성능을 향상시켰다.
- <text, audio> pair가 주어지면, Tacotron은 랜덤 초기화와 함께 처음부터 학습될 수 있다.
- Phoneme level의 alignment가 필요없기 때문에, text만 있는 많은 양의 audio를 쉽게 사용할 수 있다.
- 간단한 waveform 합성 테크닉과 함께, Tacotron은 미국 영어 평가 세트에서 3.82의 MOS를 기록했고, 이는 자연스러움 부분에서 기존의 parametric 시스템을 능가한다.


## 2 RELATED WORK

**기존 모델들의 문제점과 비교**

**waveNet**

- WaveNet은 강력한 audio 생성 모델이다.
- WaveNet은 TTS에서 높은 성능을 가지지만, 샘플 단위의 autoregressive 동작때문에 매우 느리다.
- TTS 앞단에서 언어학적 특성을 추출해 wavenet의 입력으로 사용해야 하므로, end-to-end 모델이라고 할 수 없다. TTS 시스템에서 vocoder와 acoustic 모델만 바뀐 모델이라고 할 수 있다.

**DeepVoice**

- 최근에 개발된 DeepVoice는 기존 TTS 파이프라인의 컴포넌트들을 뉴럴넷으로 대체했다. 그러나 각 컴포넌트들을 개별적으로 학습되고 이 역시 end-to-end 모델이 아니다.
wang의 논문
- 2016년 발표된 wang의 논문은 Tacotron보다 더 빨리 연구된 attention 기반의 seq2seq 모델을 사용한 end-to-end TTS 모델이다.
- 그러나 이 모델은 seq2seq 모델이 alignment를 학습하는 것을 돕기 위해 사전에 학습된 HMM aligner를 사용한다.
- 따라서 얼마나 잘 alignment가 seq2seq 모델에 의해 학습되었는지 알 수 없다.

**Char2Wav**

- Char2Wav는 characters에 학습될 수 있는 독립적으로 개발된 end-to-end 모델이다.
- 그러나 Char2Wav는 SampleRNN neural vocoder를 사용하기 전에 vocoder parameters를 예측해야하는 반면, Tacotron은 직접 raw spectrogram을 예측한다.
- 또한 Char2Wav의 seq2seq와 SampleRNN model들은 별도로 사전 학습이 필요하지만 Tacotron은 사전준비없이 학습이 이루어진다.

- 우리의 모델을 학습할 때 몇가지 트릭이 사용되며 저자들은 운율(prosody)을 해친다고 지적했다.
- 이 모델은 vocoder의 파라미터들을 예측하기 때문에 vocoder를 필요로 한다. 그러므로 이 모델은 음소(phoneme) 입력으로 학습되고 실험 결과들은 약간의 한계가 있다.


## 3 MODEL ARCHITECTURE

![tacotron1_01](/assets/img/Blog/papers/tacotron1_01.JPG)
> Figure 1: Model achitecture, 이 모델은 문자를 입력으로 받아들여 해당하는 raw spectrogram을 출력하고 그것을 Griffin-Lim reconstruction algorithm에 제공하여 음성을 합성한다.


- Tacotron의 backbone은 attention기반 seq2seq model이다.
- Figure 1은 encoder, attention-based decoder, post-processing net을 포함하는 모델 구조를 보여주고있다.
- high-level에서 Tacotron은 characters를 입력으로 받고 waveforms(파형)으로 변환되는 spectrogram 프레임을 생성한다.


### 3.1 CBHG MODULE

![tacotron1_02](/assets/img/Blog/papers/tacotron1_02.JPG)
> Figure 2: Lee et al. (2016)에 채택된 The CBHG (1-D convolution bank + highway network + bidirectional GRU) module

- CBHG는 bank of 1-D convolutional filters, highway networks, bidirectional GRU RNN으로 구성되어있다.

- CBHG는 시퀀스에서 표현(representations)을 추출하는 강력한 모듈이다.

- 입력 시퀀스(input sequence)는 먼저 $$K$$ sets의 1-D convolutional filters를 통과한다. 여기서 k번째 세트는 k너비(width)를 가지는 $$C_k$$(convolution filter)이다$$(k = 1, 2,...,K)$$.

- 이러한 필터는 로컬 및 (unigrams, bigrams, up to K-grams 모델링과 유사한)contextual information(문맥정보)를 명시적으로 모델링한다.

- convolution 출력이 함께 쌓고, local invariances를 증가시키기 위해 시간으로 max pooling한다.

- 이때 stride를 1로 사용하여 원래 time resolution을 보존한다.

- 이후 fixed-width 1-D convolutions을 통과시킨다.

- 다시 residual connection을 이용해 통과된 출력을 original input sequence와 더한다.

- Batch normalization은 매 convolution layer에서 사용된다.

- convolution의 출력은 high level 특성을 추출하기 위해 multi-layer highway network을 통과한다.

- 마지막으로, forward 문맥(context)과 backward 문맥(context)에서 sequential features을 추출하기 위해서 bidirectional GRU RNN을 사용한다.

- CBHG는 machine translation에서 영감을 받았다. machine translation과의 차이점은 non-causal convolutions, batch normalization, residual connections, stride=1 max pooling이다.

- 이러한 modification으로 generalization 효과를 향상시켰다.

### 3.2 ENCODER

- Encoder의 목표는 텍스트의 robust sequential representations을 추출하는 것이다.

- Encoder에 대한 input은 character sequence로 각각의 character는 one-hot vector로 표현되고 continuous vector로 임베딩된다.

- 그런다음 "pre-net"이라 불리는 non-linear transformations 집합을 각 임베딩에 적용한다.

- 이 작업에서 우리는 dropout과 함께 bottleneck layer를 pre-net으로 사용하여 수렴을 돕고 generalization을 개선한다.

- CBHG module은 pre-net outputs을 attention module에 사용되는 최종 encoder representation으로 변환한다.

- CBHG기반 Encoder가 overfitting을 줄일뿐만 아니라 standard multi-layer RNN encoder보다 덜 잘못발음한다는것을 발견했다.

> Table 1: Hyper-parameters와 network architectures. “conv-k-c-ReLU”는 ReLu activation과 함께 width $$k$$와 $$c$$ output channels이 있는 1-D convolution을 의미한다.

![tacotron1_03](/assets/img/Blog/papers/tacotron1_03.JPG)

### 3.3 DECODER

- 디코더로는 content 기반 tanh attention decoder를 사용한다. stateful recurrent layer는 각 디코더 time step마다 attention query를 생성한다.

- context vector와 attention RNN cell의 output은 concatenate돼서 decoder RNN의 입력으로 사용된다.

- vertical residual connections와 함께 GRU stack을 사용한다. residual connection은 빨리 convergence될 수 있도록 도와준다.

- 디코더의 타겟은 중요한 디자인 초이스다.

- 이 모델은 직접 raw spectrogram을 예측하지만, 이는 speech signal과 text 사이에 alignment를 학습하기 위해서는 매우 불필요하다.(이 작업은 실제로 seq2seq를 사용하는 동기이다)

- 이러한 불필요한 중복성(redundancy) 때문에, 모델에서 seq2seq 디코딩과 waveform 합성을 위해 다른 타겟을 사용한다.(이 작업은 실제로 seq2seq를 사용하는 동기이다)

- 수정되거나 훈련될수 있는 inversion process에 대한 prosody(운율) 정보와 sufficient intelligibility를 제공하는 한, seq2seq 타겟은 매우 압축될 수 있다.

- 80-band mel-scale spectrogram를 target으로 사용하지만, 더 적은 bands나 cepstrum과 같은 더 간결한(concise) target을 사용할 수 있다.

- seq2seq target에서 waveform로 변환하기 위해서 post-processing network를 사용한다.

<br>

- decoder targets를 예측하기 위해 simple fully-connected output layer 사용한다.

- 논문에서 발견한 중요한 trick은 multiple, non-overlapping output frames을 각 decoder step마다 예측하는 것이다.

- r frames을 한번에 예측하는 것은 총 decoder steps을 r로 나누고 model size, 학습 시간, 추론 시간을 단축한다.

- 더욱 중요한 것은 이 트릭이 attention으로부터 학습되는 alignment가 더 빨리, 더 안정적으로 측정됨으로써 convergence speech가 증가했다는 것이다.

- 이것은 인접한 음성 프레임들(speech frames)이 상관관계(correlation)가 높고 각 문자(character)가 보통 여러 프레임으로  구성되기 때문이다.

- 한번에 하나의 프레임을 방출(Emitting)하는 것은 모델이 여러번의 단계에서 동일한 입력(input) 토큰에만 집중하도록 강요한다.

- 여러 프레임을 방출(Emitting)하는 것은 attention이 학습할때 더 빨리 forward로 이동하게 만든다.

<br>

- 첫 decoder 단계는 <GO> 프레임을 나타내는 all-zero frame 상태에서 일어난다.
- <GO> frame으로 표현된 Inference 상황에서 decoder 스텝 t에서, r개의 예측의 마지막 프레임은 스텝 t+1 decoder의 입력으로 사용된다.

- 마지막 prediction을 입력으로 사용하는 것은 일반화 할 수 없는 해결책(ad-hoc choice)이다. 즉 모든 r prediction들을 사용할 수도 있다.

- 학습하는 동안은 항상 모든 r-번째 ground truth frame을 decoder에 입력한다.

- 입력 프레임은 encoder에서 사용되었던 prenet을 통과한다.

- scheduled sampling과 같은 테크닉을 사용하지 않기 때문에 pre-net에서 dropout은 모델을 일반화하는데 중요한 역할을 한다. 그것은 출력 분포의 다양한 양상을 해결하기 위해 noise source를 공급한다.

### 3.4 POST-PROCESSING NET AND WAVEFORM SYNTHESIS

- post-processing net의 task는 seq2seq target을 waveforms(파형)으로 합성할 수 있는 target으로 변환하는 것이다.

- 이 모델에서 Griffin-Lim을 합성기(synthesizer)로 사용하기 때문에, post-processing net는 linear-frequency scale로 샘플링된 스펙트럼 크기(spectral magnitude)를 예측하도록 학습한다.

- post-processing net의 다른 동기는full decoded sequence를 볼수 있다는 것이다.

- 항상 왼쪽에서 오른쪽으로 실행되는 seq2seq와 대조적으로, 그것은 각 개별 프레임에 대한 예측 오류를 바로잡기 위해 forward,backward 정보를 모두 가진다.

- 이 작업에서, 더 단순한 구조도 잘 작동할 수 있지만, post-processing net으로 CBHG module을 사용한다.

- post-processing network의 개념은 매우 general하다는 것이다.

- 이 모델은 vocoder parameters와 같은 alternative targets를 예측하거나 waveform 샘플을 직접 합성하는 WaveNet과 같은 neural vocoder로 사용할 수 있다.

- 본 논문에서는 예측된 spectrogram의 waveform을 합성하기 위해 Griffin-Lim algorithm을 사용한다.

- Griffin-Lim 알고리즘에 넣기 전에 magnitudes를 1.2제곱 하는 것은 artifacts를 감소시킨다.(harmonic enhancement 효과 때문일것이다.)

-  Griffin-Lim 알고리즘이 50회 iteration을 한 후에 합리적으로 빠르게 수렴하는 것을 관찰했다.(실제로 약 30회 반복도 충분해 보인다.)

- Griffin-Lim을 TensorFlow로 구현하였다.

- Griffin-Lim은 미분 가능하지만(훈련 가능한 가중치가 없다) 이 과정에서 우리는 그것에 대해 어떠한 loss도 부과하지 않는다.

- Griffin-Limdm은 간단하고 이미 좋은 성능을 내지만 spectrogram to waveform inverter를 빠르고 고성능으로 학습할 수 있는 점에서 선택되었다.

## 4 MODEL DETAILS

- Table 1에는 hyper-parameters와 network architectures가 나열되어 있다.

- Hann windowing, 50 ms frame length, 12.5 ms frame shift, 2048-point Fourier transform(푸리에 변환)과 함께 log magnitude spectrogram을 사용한다.

- pre-emphasis (0.97)가 도움이 되는것을 발견했다.

- 모든 실험에 4 kHz sampling rate를 사용한다. 비록 더 큰 r값(e.g. r = 5)이 더 잘 작동하지만, 이 논문에서는 MOS 결과를 위해  r = 2 (output layer reduction factor)를 사용한다.

- Adam optimizer (Kingma & Ba, 2015)를 사용하고 learning rate decay는 각각 500K, 1M, 2M global steps이후, 0.001에서 시작하여 0.0005, 0.0003, 0.0001로 줄인다.

- seq2seq decoder (mel-scale spectrogram)와 post-processing net (linear-scale spectrogram) 모두에 대해 simple $l$ 1 loss을 사용한다. 두 loss는 같은 가중치를 갖는다.

- 모든 sequence는 max length에 맞춰 padding되고 32 batch size를 사용한다.

- 패딩된 부분은 zero-padded frames로 마스킹해 사용되었다.

- 하지만 이런 방식으로 학습된 모델이 언제 emitting outputs를 멈추는지를 모르고 마지막에 반복된 소리를 만들어냈다.

- 이 문제를 해결하기 위한 한 가지 간단한 방법은 zero-padded frames을 재구성하는 것이다.

## 5 EXPERIMENTS
- 전문 여성 화자가 말하는 약 24.6시간의 음성 데이터를 포함하는 북미 영어 데이터 세트로 Tacotron을 학습하였다.

- 문장들은 text normalize 된다. 예를 들어 "16"은 "sixteen"으로 변환된다.

![tacotron1_04](/assets/img/Blog/papers/tacotron1_04.JPG)
> Figure 3: Attention alignments on a test phrase. Tacotron에서 decoder 길이는 output reduction factor(출력감소계수) r=5를 사용하기 때문에 더 짧다.

### 5.1 ABLATION ANALYSIS

- 첫번째, 우리는 vanilla seq2seq model과 비교한다.

- encoder와 decoder 모두 residual RNNs layer을 사용하며 각 layer는 256 GRU cells(LSTM을 시도했고 유사한 결과를 얻었다)을 가지고 있다.

- pre-net 또는 post-processing net를 사용하지 않았으며, decoder는 linear-scale log magnitude spectrogram를 직접 예측한다.  

- 이 모델이 alignments을 학습하고 generalize(일반화)하려면 scheduled sampling이 필요하다.

- Figure3을 보면 좋지않은 alignment를 가진다.

- Figure 3(a)는 vanilla seq2seq가 poor alignment을 학습한다는 것을 나타낸다.

- attention이 moving forward전에 많은 프레임에 대해 고착되는 경향이 있어 synthesized signal(합성신호)에서 speech intelligibility이 저하되는 원인이 된다. 그 결과 naturalness와 overall duration이 파괴된다.

- 대조적으로, Tacotron은 Figure 3(c)과 같이 깨끗하고 부드러운 alignment을 학습한다.

<br>

- 두번째, 2-layer residual GRU encoder로 대체된 CBHG encoder를 사용한 모델을 비교한다.

- 모델의 나머지는 encoder pre-net을 포함하여 전부 같다.

- Figure 3(b)와 3(c)를 비교하면, 우리는 GRU encoder에서의 alignment가 noisier 하다는 것을 알 수 있다.

- synthesized signals(합성신호)를 들어보면, noisy alignment이 종종 잘못된 발음으로 이어진다.

- CBHG encoder는 overfitting을 줄이고 길고 complex phrases(복잡한 구문)을 generalize한다.

![tacotron1_05](/assets/img/Blog/papers/tacotron1_05.JPG)
> Figure 4: post-processing net을 사용한 Predicted spectrograms와 사용하지 않은 Predicted spectrograms이다.


- Figures 4는 post-processing net 사용의 장점을 보여준다.

- 다른 모든 컴포넌트들은 그대로 두고 post-processing net만 없는 모델을 훈련시켰다.(decoder RNN이 linear-scale spectrogram을 예측한다는 점을 제외)

- 더 많은 contextual information(상황별 정보)를 통해, post-processing net로부터 만들어진 prediction은 harmonics와 synthesis artifacts를 감소기키는 high frequency formant structure를 더 잘 생성한다.

### 5.2 MEAN OPINION SCORE TESTS

- 본 논문에서는 mean opinion score test를 진행했고, 5 point의 리커트 척도를 사용해 자연스러움에 대해 물어봤다.

- MOS 테스트는 원어민으로부터 크라우드소싱 방식으로 모아 평가했다.

- 100개의 학습되지 않은 문장들은 테스트를 위해 사용되었고 각 문장은 총 8번의 점수를 받았다.

- MOS를 계산할 때 오직 헤드폰을 사용해서 평가한 결과만 사용하였다.

- 제안하는 모델을 실제 사용되는 parametric 모델과 concatenative 모델과 비교하였다.

- Tacotron의 MOS는 3.82로 parametric 모델 이상의 성능을 가진다.

- 강력한 baseline과 Griffin-Lim 합성에 의해 발생한 artifacts를 고려한다면, 이 결과는 매우 괜찮은 결과다.

> Table 2: 5-scale mean opinion score evaluation

![tacotron1_06](/assets/img/Blog/papers/tacotron1_06.JPG)

## 6 DISCUSSIONS

- 본 논문에서는 Tacotron을 제안했고 이는 character 시퀀스를 입력으로 받아 spectrogram을 출력하는 end-to-end TTS 모델이다.

- 간단한 waveform 합성기 모듈과 함께 3.82의 MOS에 도달하였고, 실제 자연스러움의 부분에 있어 parametric system의 성능을 능가하였다.

- Tacotron은 frame-based라서 inference가 sample-level의 autoregressive 모델에 비해 빠르다.

- 이전 연구와 달리, Tacotron은 손이 많이 가는 언어학적 특성 추출 또는 HMM aligner와 같은 복잡한 컴포넌트들이 필요하지 않다.

- 이 모델은 단순히 랜덤초기화와 함께 처음부터 학습될 수 있다.

- 학습된 text normalization에서의 최근 발전이 미래에는 text normalization을 불필요하게 할수 있겠지만, 이 모델에서는 simple text normalization을 사용하였다.

- 우리는 아직 우리의 모델을 많은 측면에서 더 조사해야한다. 많은 초기 모델을 바꾸지 않은 상태다.

- Output layer, attention module, loss function, Griffin-Lim-based waveform synthesizer는 모두 개선 가능성이 있다.

- 예를 들어, Griffin-Lim 출력은 audible artifacts를 가진다고 널리 알려져 있다.

- 우리는 현재 빠르고 높은 퀄리티의 neural-network-based spectrogram inversion을 연구하고 있다.
