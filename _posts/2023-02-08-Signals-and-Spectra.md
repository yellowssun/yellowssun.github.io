---
marp: true
math: katex
paginate: true
backgroundColor: #fff
title: Signal and Spectra
categories:
  - Digital Communications
toc: true
toc_sticky: true
tags:
  - [Sklar, Signal, Spectra]
date: 2023-02-08
last_modified_at: 2023-02-08
---

## 1. Digital communication Signal Processing

* 1. 1 Why Digital?
            *아날로그 통신 방식에서 디지털 통신 방식으로 변화하는 여러 이유들이 존재한다. 첫 째로는 모든 전송선과 회로에는 주파수 전송과 같은 왜곡 현상이 발생한다. 두번째로는 전기적인 잡음 및 간섭 현상으로 인한 왜곡이다. 디지털 증폭기는 이와 같은 왜곡들을 이상적인 펄스 형태로 다시 복원할 수 있다. **즉, 다시 말해 디지털 회로는 아날로그 회로에 비해 왜곡 및 간섭에 강하다.** 아날로그 신호는 무한한 다양한 형태의 신호로 존재하여 약간의 간섭이 복원 불가능할 정도로 큰 왜곡이 될 수 있다.
            * 왜곡에 강건하다는 특징 이외에도 다른 장점들이 있다. 디지털 회로는 저렴하고 신뢰할 수 있다. 디지털 하드웨어 구현이 용이하기 때문에 광범위하게 활용될 수 있다.
              EX. 디지털 신호를 사용하는 time division multiplexing 이 안날로그 신호를 사용하는 frequency division multiplexing 보다 쉽게 합성된다.
            * 디지털 통신 시스템에서 대부분의 자원은 신호의 싱크를 맞추는데 할당된다.  이는 10장에서 다시 다룬다. 디지털 통신 시스템에서 SNR이 임계값까지 떨어진 경우 QoS 는 급격하게 좋아지거나 나빠질 수 있다. 이러한 단점을 Nongraceful degradation라 한다. 대조적으로 대부분의 아날로그 통신 시스템은 유연하게 변화한다.
        * 1. 2 Typical Block Diagram and Transformations
            * 디지털 통신의 로드맵은 아래 사진과 같다. ![Block diagram of a typical digital communication system](https://dynalist.io/u/9-Q1HseWpCc2KAwfvihRPVdo)
            * 상위 블록은 format, source encode, encrypt, channel encode, multiplex, pulse modulate, bandpass modulate, frequency spread, and multiple access으로 이루어져 있고 이는 신호를 전송하는 데 필요한 요소들이다. 하위 블록은 신호를 받기 위해 상위 블록의 역방향의 형태로 이루어져 있다.
            * Formatting-
              `Character coding`, `sampling`, `Quantization`, `Pulse code modulation(PCM)`
            * Source Coding- 아날로그 신호를 디지털 신호로 변화시킨 후 redundant 정보를 제거한다.
              `Predictive coding`, `Block coding`, `Variable length coding`, `Synthesis/analysis coding`, `Lossless compare
            * Baseband Signaling-
              `PCM waveforms` (Nonreturn-to-zero, Return-to-zero, Phase encoded, Multilevel binary), `M-ary pulse modulation` (PAM, PPM, PDM)
            * Equalization-
              `Maximum-likelihood sequence estimation (MLSE)`, `Equalization with filters` (Transversal or decision feedback, Preset or Adaptive, Symbol spaced or fractionally spaced)
            * Bandpass Signaling(coherent)-
              `Phase shift keying(PSK)`, `Frequency shift keying(FSK)`, `Amplitude shift keying(ASK)`, `Continuous phase modulation(CPM)`, `Hybrids`
            * Bandpass Signaling(noncoherent)-
              `Differential phase shifting keying(DPSK)`, `Frequency shift keying(FSK)`, `Amplitude shift keying(ASK)`, `Continuous phase modulation(CPM)`, `Hybrids`
            * Channel Coding(waveforms)-
              `M-ary signaling`, `Antipodal`, `Orthogonal`, `Trellis-coded modulation`
            * Channel Coding(Structured Sequences)-
              `Block`, `Convolutional`, `Turbo`
            * Synchronization-
              `Frequency synchronization`, `Phase synchronization`, `Symbol synchronization`, `Frame synchronization`, `Network synchronization`
            * Multiplexing/Multiple Access-
              `Frequency division (FDM/FDMA)`, `Time division (TDM/TDMA)`, `Code division (CDM, CDMA)`, `Space division (SDMA)`, `Polarization division (PDMA)`
            * Spreading-
              `Direct sequencing (DS)`, `Frequency hopping (FH)`, `Time hopping (TH)`, `Hybrids`
            * Encryption-
               `Block`, `Data stream`
        * 1. 3 Basic Digital Communication Nomenclature
            * Information source- 원 정보는 아날로그와 디지털의 형태로 존재한다. 아날로그 정보의 출력은 연속적인 진폭을 갖고 디지털 정보의 출력은 유한한 진폭을 갖는다. 아날로그 정보는 sampling과 quantizatio 과정을 통해 디지털 정보로 변환될 수 있다.
            * Textual message- 디지털 송신에서 문자는 유한한 디지털 기호 및 숫자로 나타낼 수 있다.
            * Character- Character는 이진수의 배열로 나타낼 수 있다. 대표적인 예로 American Standard Code for Information Interchange (ASCII), Extended Binary Coded Decimal Interchange Code (EBCDIC), Hollerith, Baudot, Murray, 그리고 Morse가 있다.
            * Binary digit- 디지털 시스템의 근본적인 정보 단위이다. [bit]
            * Bit stream- 0과 1로 이루어진 이진 배열로 baseband signal에서 주로 사용되는 용어이다.  ![Bit stream (7-bit ASCII)](https://dynalist.io/u/mxE-jLKpgBcbbRiNOz13g8RV). 실제 시스템에선 이와같은 공간을 유용하게 사용할 수 없기 때문에 사진과 같은 펄스는 존재하지 않는다.
            * Symbol
            * Digital waveform
            * Data rate
        * 1. 4 Digital versus Analog Performance Criteria
            * 디지털과 아날로그를 비교하기 위한 기준으로 SNR, 왜곡 비율, mse 기대값을 주로 사용한다.

## 2. Classification of Signals

        * 2. 1 Deterministic and Random Signals
            * 신호는 결정적, 불규칙적으로 분류 할 수 있다. 결정론적 신호는 수학적인 표현으로 모델을 구현할 수 있다. 불규칙 신호는 수식적으로 명쾌하게 표현할 수 없지만 충분히 긴 시간에서 규칙성을 찾아 확률적, 통계적으로 표현할 수 있다. 
        * 2. 2 Periodic and Nonperiodic Signals
            * 신호는 주기 신호와 비 주기 신호로 분류 할 수 있다. 다음과 같은 수식을 만족하지 않는 다면 비 주기 신호로 분류한다.

$$x(t)=x(t+T_0), for -\infty<t<\infty$$
        * 2. 3 Analog and Discrete Signals
            * 아날로그 신호는 시간에 연속적인 함수를 갖고 모든 시간에 유일하게 정의된다. 이산 신호는 이산 시간에서만 값이 존재한다.
        * 2. 4 Energy and Power Signals
            * 통신 신호를 분석할 때 주로 파장 에너지를 다룬다.무한한 시간에서의 에너지가 유한한 신호를 energy signal로 정의한다. Energy signal을 수식적으로 표현하면 다음과 같다. 
$$E_x=\lim_{T\to \infty}\int_{-T/2}^{T/2}{x^2}(t)dt=\int_{-T/2}^{T/2}{x^2}(t)dt$$
            * 실생활에서의송신 신호는 유한한 에너지를 갖는다. 그러나 주기 신호는 무한한 시간에서 정의되기 때문에 무한한 에너지를 갖는다. 이러한 주기 신호를 다시 power signal 로 정의한다. Power signal을 수식적으로 표현하면 다음과 같다.
$$P_x=\lim_{T\to \infty}\frac{1}{T}\int_{-T/2}^{T/2}{x^2}(t)dt$$
            * Energy signal과 Power signal은 서로 베타적으로 energy signal은 유한한 에너지와 평균 0의  power를 갖고, power signal은 유한한 power와 무한한 에너지 값을 값는다.
        * 2. 5 The Unit Impulse Function
            * 통신 이론에서 impulse or Dirac delta 함수는 중요하게 사용된다. Impulse 함수의 수식은 다음과 같다.
$$\int_ {-\infty}^\infty \delta(t)\mathrm{d}t=1$$
$$\delta(t)=0, for $$  $$t\ne 0$$
$$\delta(t)$$ $$is$$ $$unbounded$$ $$at$$ $$t\ne0$$
$$\int_{-\infty}^\infty x(t)\delta(t-t_0)dt=x(t_0)$$

## 3. Spectral Density

        * 통신 시스템의 필터링 과정에서 spectral density는 중요한 개념이다. 필터의 출력에서 신호와 잡음을 평가해야 하는데 Energy Spectral Density (ESD) 와 Power Spectral Density (PSD) 는 이때 평가 지표로 사용된다.
        * 3. 1 Energy Spectral Density @ESD
            * Non-periodic signal $$x(t)$$는 푸리에 변환을 통해 $$X(f)$$로 표현될 수 있다. ESD는 주파수 함수로써의 에너지 분포로 __@Parseval's theorem__을 통해 시간 영역에서 구한 총 에너지와 동일 함을 알 수 있다.
            * $$Total\;Energy: E_x=\int_{-\infty}^{\infty}{x^2}(t)dt=\int_{-\infty}^\infty|X(f)|df$$
            * 추가적으로 __@Wiener-Khinchine Theorem__를 통해 Autocorrelation function의 Fourier Transform은 ESD와 동일 함을 알 수 있다. 즉, 주파수 영역 전제를 적분하면 총 에너지가 된다.
            * $$Autocorrelation function: R_x(\tau)=\int_{-\infty}^{\infty}{x}(t){x}(t+\tau)dt$$
            * $$F[R_x(\tau)]=F[\int_{-\infty}^{\infty}{x}(t){x}(t+\tau)dt]=E_x$$
            * 단, random process 에서 ESD는 정의할 수 없다.
        * 3. 2 Power Spectral Density @PSD
            * Periodic signal 은 Fourier series 를 적용하여 표현할 수 있다. $$c_n$$ 은 Fourier coefficient 를 의미한다. 푸리에 급수로 전개 후 power spectral density는 $$G_x(f)$$로 나타낸다. 수식은 다음과 같다.
            * $$Average Power: P_x=\frac{1}{T}\int_{-T/2}^{T/2}{x^2}(t)dt=\sum_{n=-\infty}^\infty{|c_n|^2}$$
            * $$PSD: G_x(f)=\sum_{n=-\infty}^\infty{|c_n|^2}\delta(f-nf_0)$$
            * $$P_x=\int_{-\infty}^{\infty}{G_x}(f)df=2\int_{0}^\infty{G_x}(f)df$$
    
    
## 4. Autocorrelation

        * 4. 1 Autocorrelation of an Energy Signal
            * Autocorrelation은 신호가 지연된 상태와 지연되지 않은 원신호 간의 불일치 성을 보여준다. 즉, $$R_x(\tau)$$는 얼마나 원신호와 유사한지 보여주는 척도 이다. Autocorrelation function 을 실수 집합에서 정의하면 다음과 같다.
            * $$R_x(\tau)=\int_{-\infty}^\infty x(t)x(t+\tau)dt$$ $$for$$ $$-\infty<\tau<\infty$$
            * 중요한 점은 auctocorrelation function 은 시간의 함수가 아닌 시간 차이 즉  $$\tau$$의 함수라는 점이다.
            * $$R_x(\tau)$$의 특징으로는 다음과 같다.
                * $$R_x(\tau)=R_x(-\tau)$$
                  symmetrical in $$\tau$$ about zero (우함수)
                * $$R_x(\tau)\le R_x(0)$$ $$for$$ $$all$$ $$\tau$$
                  maximum value occurs at the origins (시간 차이가 없는 경우 가장 큰 값)
                * $$R_x(\tau)\leftrightarrow\Psi(f)$$
                  autocorrelation and ESD form a Fourier transform pair, as designated by the double-headed arrows
                * $$R_x(0)=\int_{-\infty}^\infty {x^2}(t)dt$$
                  value at the origin is equal to the energy of the signal
        * 4. 2 Autocorrelation of a Periodic (Power) Signal
            * Power signal의 실수 집합에서의 autocorrelation function 은 다음과 같다.
            * $$R_x(\tau)=\lim_{T \to \infty}{\frac{1}{T}}\int_{-\infty}^\infty x(t)x(t+\tau)dt$$, $$for$$ $$-\infty<\tau<\infty$$
            * 주기 신호의 주기를 $$T_0$$ 로 설정하고 수식에 대입한다면 다음과 같다.
            * $$R_x(\tau)={\frac{1}{T_0}}\int_{-\frac{T_0}{2}}^\frac{T_0}{2} x(t)x(t+\tau)dt$$ $$for$$ $$-\infty<\tau<\infty$$
            * 주기 신호의 autocorrelation 함수의 특징은 다음과 같다.
                * $$R_x(\tau)=R_x(-\tau)$$
                  symmetrical in $$\tau$$ about zero (우함수)
                * $$R_x(\tau)\le R_x(0)$$ $$for$$ $$all$$ $$\tau$$
                  maximum value occurs at the origin
                * $$R_x(\tau)\leftrightarrow G_x(f)$$
                  autocorrelation and PSD form a Fourier transform pair
                * $$R_x(0)=\frac{1}{T_0}\int_{-\frac{T_0}{2}}^\frac{T_0}{2} {x^2}(t)dt$$
                  value at the origin is equal to the average power of the signal
    
## 5. Random Signals

        * 통신 시스템의 주된 목적은 정보를 채널을 통해 보내는 것이다. 대부분의 신호들은 불규칙적으로 나타나기 때문에 수신기는 어떤 파장의 신호가 올지 모른다. 따라서 불규칙 신호의 형태를 수식적으로 표현할 수 있어야 한다.
        * 5. 1 Random Variables
          Random variables 는 이산적인 확률변수, 연속적인 확률변수로 나눌 수 있다. 본 장에서는 연속적인 확률변수 (Continuous Random Variable) 을 중점적으로 다룬다.
            * $$Cumulative\;Distribution\;Function (CDF):$$
                * $$ F_X(x)=P(X\le x)$$
                * Properties: Non-decreasing function
                    * $$0\le F_X(x)\le 1$$
                    * $$F_X(x_1)\le F_X(x_2),\;if\; x_1\le x_2$$
                    * $$F_X(-\infty)=0$$
                    * $$F_X(\infty)=1$$
            * $$probability\;density\;function: p_X{(x)}=\frac{dF_X(x)}{dx}$$
                * Properties: Non-negative function
                    * $$p_X{(x)}\ge 0$$
                    * $$\int_{-\infty}^\infty p_X (x)dx=1$$
                    * $$P[a<X\le b]=P[X\le b]-P[X\le a]$$ 
                                  $$=F_X(b)-F_X(a)=\int_{a}^b p_X (x)dx$$
                    * $$\int_{-\infty}^\infty p_X{(x)}dx=F_X(\infty)-F_X(-\infty)=1$$
            * Ensemble Averages
              (고정된) 시간 t에서 random process $$X(t)$$의 기대값
                * mean value: $$m_X$$ or expected value of random variable $$X$$
$$m_X=E[X]=\int_{-\infty}^\infty x*p_X (x)dx$$
                * $$n^{th}$$ moment의 확률 분포:
$$E[X^n]=\int_{-\infty}^\infty x^n p_X (x)dx$$
                * 통신 시스템에서는 1, 2번째 moment가 중요하다. 1번째 moment와 2번째 moment 를 통해 mean (평균), variance (분산), standard deviation (표준편차) 을 구할 수 있다.
                * $$Variance:$$ 
$$var(X)=E[(X-m_X)^2]=\int_{-\infty}^\infty (x-m_X)^2 p_X (x)dx$$
                * $$Standard \;deviation:$$
$$\sigma^2_X=E[X^2-2m_X X+m^2_X]=E[X^2]-m^2_X$$
        * 5. 2 Random Processes
            * Random process는 사건과 시간 두 개의 확률 변수로 나타낼 수 있다. 동일한 generator 에서 신호가 출력되도 각각의 신호는 다른 잡음, 간섭이 일어나기 때문에 각각의 신호는 다른 형태를 띄고 있다. 
시간에 따른 N개의 sample: $$X_j (t)$$  ![Random noise process](https://dynalist.io/u/krZarZoFKyD24_kMRKDM6lU7)
            *Statistical Averages of a Random Process
                * Random process 의 미래 시간은 알 수 없기 때문에 통계적으로 **pdf** 를 표현.
                *$$E\{X(t_k)\}=\int^{\infty}_{-\infty}x*p_{X_k}(x)dx=m_X(t_k)$$
                * $$Autocorrelation\;function: R_X (t_1, t_2)= E\{X(t_1)X(t_2)\}$$
            * Stationarity
              통계적 성질이 시간에 따라 변하지 않음
              여러 시간 구간 마다 동일한 통계적 특성, 모든 시간에서 동일한 성질을 갖는 확률변수 관측됨
                * 대부분의 통신 이론에서 불규칙 신호와 잡음은 모든 시간 내에서 wide-sense stationary 임을 가정한다.
                * $$WSS: E\{X(t)\}=m_X=constant$$
                  $$R_X(t_1, t_2)=R_X(t_1-t_2)$$
            * Autocorrelation of a Wide-Sense Stationary (WSS) Random Process
                * Autocorrelation: $$R_X(\tau)=E[X(t)X(t+\tau)]\; for -\infty<\tau<\infty$$
WSS process에서 autocorrelation 함수는 $$\tau =t_1-t_2$$의 함수이다.
                * WSS: 평균이 상수이고 autocorrelation 함수가 $$\tau$$의 함수인 process를 의미한다.
                * $$R_X (\tau)=E[X(t)X(t+\tau)]$$
$$P=R_X(0)=E[X^2 (t)]$$
                * Properties of WSS process
                    * $$R_X(\tau)=R_X(-\tau)$$
                      symmetrical in $$\tau$$ about zero
                    * $$R_X(\tau)\le R_X(0)\; for\; all \; \tau$$
                      maximum value occurs at the origin
                    * $$R_X(\tau)\leftrightarrow G_X(f)$$
                      autocorrelation and power spectral density form a Fourier transform pair
                    * $$R_X(0)=E[X^2 (t)]$$
                      value at the origin is equal to the average power of the signal
        * 5. 3 Time Averaging and Ergodicity
            * Ensemble averaging 을 통해 $$m_X$$ 와 $$autocorrelation function$$ 을 구하기 어렵기 때문에 어떤 한 불규칙 신호를 ergodic 하다라고 가정한다. 어떤 한 random process 가 ergodic 하다는 것은 strict stationary 를 만족한다는 것이다.
              Strict stationary는 WSS의 상위 범주로 통계적 성질이 시간에 따라 변하지 않는 다는 특징을 가지고 있다.
            * Ergodic process: 어떤 함수에 대해서도 ensemble 평균이 시간 평균과 동일한 경우. 통계적 특징은 하나의 sample function의 시간 평균에 따라 결정된다. Ergodic process의 조건식은 다음과 같다.
            * $$m_X=\lim_{T\to \infty}\frac{1}{T}\int_{-T/2}^{T/2}X(t)dt$$
$$R_X(\tau)=\lim_{T\to \infty}\frac{1}{T}\int_{-T/2}^{T/2}X(t)X(t+\tau)dt$$
            * Ergodic process를 실제로 테스트하는 것은 어렵다. 통신 신호를 분석함에 있어 합리적인 가정은 불규칙 파장을 ergodic 하다 가정하는 것이다. Ergodic process에서 시간적 평균이 ensemble 평균과 같다는 것은 DC, rms, average power value가 ergodic 확률 변수와 관련있을 수 있다는 것이다.
                * 1. The quantity $$m_X=E\{X(t)\}$$ is equal to the dc level of the signal.
                * 2. The quantity $$m^2_X$$ is equal to the normalized power in the dc component.
                * 3. The second moment of $$X(t), E\{X^2(t)\}$$, is equal to the total average normalized power.
                * 4. The quantity $$\sqrt{E\{X^2(t)\}}$$ is equalt to the root-mean-square (rms) value of the voltage or current signal.
                * 5. The variance $$\sigma^2_X$$ is equal to the average nromalized power in the time-varying or ac component of the signal.
                * 6. If the process has zero mean ($$i.e., m_X=m^2_X=0$$),
then $$\sigma^2_X=E\{X^2\}$$ and the variance is the same as the mean-square value, or the variance represents the total power in the normalized load.
                * 7. The standard deviation $$\sigma_X$$ is the rms value of the ac component of the signal.
                * 8. If $$m_X=0$$, then $$\sigma_X$$ is the rms value of the signal.
        * 5. 4 Power Spectral Density of a Random Process
          Non-periodic signal (noise 추가)
            * PSD 는 네트워크를 통과하는 신호 전력을 주파수 관점에서 평가할 수 있다.
            *  ![Autocorrelation and power spectral density 1](https://dynalist.io/u/J97lQnLAWYRUOefcKDLGMbch)
 ![Autocorrelation and power spectral density 2](https://dynalist.io/u/GU5NTsxE5LiQ4R8ZJfq2TYCZ)
 ![Autocorrelation and power spectral density 3](https://dynalist.io/u/uoXDF4KCX9ihUf5by9SqNIwp)
 ![Autocorrelation and power spectral density 4](https://dynalist.io/u/UMXkwYFc5nD8BGid9Q-H1CHy)
            * Random process의 PSD는 다음과 같다.
                * $$G_X(f)\ge 0$$
                  always real valued
                * $$G_X (f) =G_X (-f), \; for\; X(t) \; real-valued$$
                  $$X(t)$$ real-valued
                * $$G_X(f)\leftrightarrow R_X(\tau)$$
                  PSD and autocorrelation form a Fourier transform pair
                * $$P_X=\int_{-\infty}^{\infty}{G_X (f)}df$$
                  relationship between average normalized power and PSD

            * PSD의 적분은 평균 전력을 의미한다. 대역폭을 측정하는 간단한 방법은 main spectral lobe 를 확인하는 것이다. PSD의 autocorrelation function 의 main spectral lobe를 통해 간단하게 대역폭을 확인할 수 있다. (정확한 값은 아님)
        * 5. 5 Noise in Communication System
            * 전기 신호에서 잡음연 여러 형태로 존재한다. 효과적인 성능을 보이기 위해선 이러한 잡음을 filtering, shielding, modulation 과 같은 방법으로 줄여야한다. 자연에 존재하는 대표적인 잡음은 열 잡음이다. 이를 수식적으로 표현하면 다음과 같다.
            * $$p(n)=\frac{1}{\sigma \sqrt{2\pi}} exp[-\frac{1}{2}(\frac{n}{\sigma})^2]$$
            * 열 잡음은 평균이 0인 Gaussian 확률 분포를 따른다. DC 신호에서의 Gaussian 잡음의 확률 변수는 다음과 같다.
            * $$z=a+n; (a: message, n: noise)$$
            * White Noise
                * 열 잡음의 특징으로는 power spectral density가 모든 주파수에서 동일하다는 점이다. 즉, 열 잡음은 모든 주파수 영역에서 대역폭에 동일한 잡음 전력을 내뿜는다. 
                * $$G_n(f)=\frac{N_0}{2}, watts/herz \rightarrow$$ noise PSD는 flat하다.
                * $$R_n(\tau)=F^{-1}\{G_n(f)\}=\frac{N_0}{2}\delta(\tau)$$
White noise의 auctocorrelation 은 delta function 으로 표현 가능하다. White noise의 대역폭은 무한하기 때문에
$$P_n=\int^{\infty}_{-\infty}\frac{N_0}{2}df=\infty$$ 이 성립한다.
                * 실제 process 에서 white noise 는 존재하지 않지만 대부분의 실제 시스템에 whtie 를 가정하여 적용한다. 잡음의 대역폭이 시스템의 대역폭보다 큼에 따라 잡음의 대역폭은 무한하다고 가정 할 수 있다.
                * Properties:
Additive, White, Gaussian $$\rightarrow$$ Addaptive White Gaussian Noise (AWGN)
    
## 6. Signal Transmission through Linear Systems

        * Linear-Time Invariant (LTI) system:  ![Linear system](https://dynalist.io/u/kbUWyKIvnAotuuQWBVv4Z2NH)
        * 6. 1 Impulse Response
            * $$h(t)=y(t)\;\leftarrow when\; x(t)=\delta(t) \;(impulse\;response)$$
            * $$y(t)=x(t)*h(t)=\int^{\infty}_{-\infty}x(\tau)h(t-\tau)d\tau$$
            *$$If\;the\;system\;is\;casual \rightarrow$$
$$y(t)=x(t)*h(t)=\int^{\infty}_0x(\tau)h(t-\tau)d\tau$$
        * 6. 2 Frequency Transfer Function
            * $$Y(f)=X(f)H(f)\;or\;H(f)=\frac{Y(f)}{X(f)}$$
$$H(f): frequency\;transfer\;function$$
            * $$H(f)=|H(f)|e^{j\theta f}$$
$$\theta(f)=tan^{-1}\frac{Im\{H(f)\}}{Re\{H(f)\}}$$
            * Random Processes and Linear System
                * LTI system:
$$Input (random process) \rightarrow Output (random process)$$
$$Input (sample function) \rightarrow Output (sample function)$$
$$Input (power spectral density) \rightarrow Output (power spectral density)$$
        * 6. 3 Distortionless Transmission
            * 이상적인 무왜곡 전송은 다음과 같은 신호로 표현될 수 있다.
            * $$y(t)=Kx(t-t_0)\rightarrow Y(f)=KX(f)e^{-j2\pi f t_0}$$
$$H(f)=Ke^{-j2\pi f t_0}$$
            * 즉, 이상적인 무왜곡 전송을 얻기 위해서 시스템 응답은 상수의 크기를 갖고 phase shift는 주파수에서 선형적이어야 한다(?).
            * Ideal Filter
                * 무한대의 대역폭을 갖는다고 가정
                * 왜곡 없이 truncated network that passes
                * all frequency components between $$f_l$$ and $$f_u$$ (범위가 정확하게 나눠짐)
            * Realizable Filters
                * $$\scrt{F}$$
        * 6. 4 Signals, Circuits, and Spectra
          신호가 필터 회로를 지나가면 신호의 대역폭은 어떻게 변할까?
            * 신호는 spectra 로 표현될 수 있다. Network 혹은 circuit은 spectral characteristics 혹은 frequency transfer function 으로 설명할 수  있다.
            * 주파수 관점에서 입력 신호의 대역폭이 채널 대역폭보다 좁다면 출력 신호의 대역폭은 입력 신호의 대역폭으로 제한된다. 반대의 경우도 마찬가지이다.
            * ==입력 신호의 대역폭과 채널 대역폭의 길이에 따라 변화한다. ==

## 7. Bandwidth of Digital Data

        * 7. 1 Baseband versus Bandpass
            * Low-pass or baseband 신호를 고주파수로 변화시키기 편한 방법은 basebad signal 에 carrier wave ($$cos2\pi f_ct$$) 을 담아 곱하는 것이다. 이렇게 구한 신호 ($$x_c(t)$$) 는 double-sidedband (DSB) modulated signal 이라 부른다.
$$x_c(t)=x(t)cos(2\pi f_c t)\rightarrow \; X_c(f)=\frac{1}{2}[X(f-f_c)+X(f+f_c)]$$
 ![Heteridyning](https://dynalist.io/u/ZDODbzj5dFHZoRjvLUyYG8VJ)
        * 7. 2 The Bandwidth Dilemma
            *
## 8. Conclusion

        * Chapter 1 에서는 본 책의 개요와 기초 명명법을 정의하였다. 시변 신호의 classification, spectral density, autocorrelation 을 알아보았다. 또한 대부분의 통신 시스템에 잡음 성분이 더해진 불규칙 신호의 특징들을 알아보았다. 마지막으로 선형 시스템의 신호 전송의 중요한 분야를 다루고 이상적인 case 를  적용하였다.
