<div align="center">
<img src="logo.png" width="150">
</div>

# Awesome Machine Learning in Plasma Physics [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A curated list of awesome machine learning resources for plasma physics, tokamaks, and stellarators.

## Contents

[Papers 📚](#research-papers) | [Implementations 💻](#implementation-papers) | [Tools 🛠️](#tools)

## Research Papers

*Papers are organized chronologically*

**Format:** "**title** - **year** - *authors* - journal/conference/thesis - link - abstract or summary"

**A generative artificial intelligence framework for long-time
               plasma turbulence simulations** - **2025** - *Clavier, B, Zarzoso, D, del-Castillo-Negrete, D and
               Fr\'{e}nod, E* - journal - [arXiv](https://arxiv.org/abs/https://pubs.aip.org/aip/pop/article-pdf/doi/10.1063/5.0255386/20553361/063905\_1\_5.0255386.pdf) | [DOI](https://doi.org/10.1063/5.0255386) - Generative deep learning techniques are employed in a novel
               framework for the construction of surrogate models capturing the
               spatiotemporal dynamics of 2D plasma turbulence. The proposed
               Generative Artificial Intelligence Turbulence (GAIT) framework
               enables the acceleration of turbulence simulations for long-time
               transport studies. GAIT leverages a convolutional variational
               auto-encoder and a recurrent n... <!-- imported-from-bib -->

**Active ramp-down control and trajectory design for tokamaks with
               neural differential equations and reinforcement learning** - **2025** - *Wang, Allen M, Rea, Cristina, So, Oswin, Dawson, Charles, Garnier, Darren T, Fan, Chuchu* - journal - [DOI](https://doi.org/10.1038/s42005-025-02146-6) - The tokamak offers a promising path to fusion energy, but
               disruptions pose a major economic risk, motivating solutions to
               manage their consequence. This work develops a reinforcement
               learning approach to this problem by training a policy to
               ramp-down the plasma current while avoiding limits on a number of
               quantities correlated with disruptions. The policy training
               environment is a hybrid physics and mach... <!-- imported-from-bib -->

**Impact of model uncertainty on {SPARC} operating scenario
                   predictions with empirical modeling** - **2025** - *Saltzman, A, Rodriguez-Fernandez, P, Body, T, Ho, A, Howard, N T* - journal - [arXiv](https://arxiv.org/abs/2506.09879) - Understanding and accounting for uncertainty helps to ensure
                   next-step tokamaks such as SPARC will robustly achieve their
                   goals. While traditional Plasma OPerating CONtour (POPCON)
                   analyses guide design, they often overlook the significant
                   impact of uncertainties in scaling laws, plasma profiles, and
                   impurity concentrations on performance predictions. This work
                   confronts these cha... <!-- imported-from-bib -->

**Magnetic control of tokamak plasmas through deep reinforcement
              learning with privileged information** - **2025** - *Sorokin, Dmitri, Granovskiy, Aleksandr, Kharitonov, Ivan and
              Stokolesov, Maksim, Prokofyev, Igor, Adishchev, Evgeny and
              Subbotin, Georgy, Nurgaliev, Maxim* - journal - [OpenReview](https://openreview.net/forum?id=vp6K02lz4E) - Reinforcement learning (RL) is capable of training
              high-performance control policies for a variety of domains from
              computer games [1] to physical robots [2, 3] and scientific
              equipment [4]. Recently, RL was applied to train real-time
              controllers for a tokamak plasma [5, 6, 7]. Tokamak plasma is
              controlled at a time scale of hundreds of microseconds and
              requires precise stabilization algorithms, which should be... <!-- imported-from-bib -->

**Physics-informed neural networks for the modelling of
               interferometer-polarimetry in tokamak multi-diagnostic
               equilibrium reconstructions** - **2025** - *Rutigliano, Novella, Rossi, Riccardo, Murari, Andrea and
               Gelfusa, Michela, Craciunescu, Teddy, Mazon, Didier and
               Gaudio, Pasquale* - journal - [DOI](https://doi.org/10.1088/1361-6587/addde6) - Abstract Equilibrium reconstruction is crucial in nuclear fusion
               and plasma physics, as it enables the understanding of the
               distribution of fundamental plasma quantities within a reactor.
               Given that equilibrium reconstruction is an ill-posed problem, it
               is essential to constrain the algorithm with multiple diagnostics
               to achieve accurate results. Among these, the
               interferometer-polarimeter is one of the ... <!-- imported-from-bib -->

**Reconstructing the plasma boundary with a reduced set of
                   diagnostics** - **2025** - *Stokolesov, M S, Nurgaliev, M R, Kharitonov, I P and
                   Adishchev, E V, Sorokin, D I, Clark, R, Orlov, D M* - journal - [arXiv](https://arxiv.org/abs/2505.10709) - This study investigates the feasibility of reconstructing the
                   last closed flux surface (LCFS) in the DIII-D tokamak using
                   neural network models trained on reduced input feature sets,
                   addressing an ill-posed task. Two models are compared: one
                   trained solely on coil currents and another incorporating
                   coil currents, plasma current, and loop voltage. The model
                   trained exclusively on c... <!-- imported-from-bib -->

**Reconstruction-free magnetic control of {DIII-D} plasma with
                   deep reinforcement learning** - **2025** - *Subbotin, G F, Sorokin, D I, Nurgaliev, M R and
                   Granovskiy, A A, Kharitonov, I P, Adishchev, E V and
                   Khairutdinov, E N, Clark, R, Shen, H, Choi, W and
                   Barr, J, Orlov, D M* - journal - [arXiv](https://arxiv.org/abs/2506.13267) - Precise control of plasma shape and position is essential for
                   stable tokamak operation and achieving commercial fusion
                   energy. Traditional control methods rely on equilibrium
                   reconstruction and linearized models, limiting adaptability
                   and real-time performance. Here,the first application of deep
                   reinforcement learning (RL) for magnetic plasma control on
                   the mid-size DIII-D tokamak... <!-- imported-from-bib -->

**A generative machine learning surrogate model of plasma
                   turbulence** - **2024** - *Clavier, B, Zarzoso, D, del-Castillo-Negrete, D and
                   Frenod, E* - journal - [arXiv](https://arxiv.org/abs/2405.13232) - Generative artificial intelligence methods are employed for
                   the first time to construct a surrogate model for plasma
                   turbulence that enables long time transport simulations. The
                   proposed GAIT (Generative Artificial Intelligence Turbulence)
                   model is based on the coupling of a convolutional variational
                   auto-encoder, that encodes precomputed turbulence data into a
                   reduced latent spac... <!-- imported-from-bib -->

**A high-density and high-confinement tokamak plasma regime for
               fusion energy** - **2024** - *Ding, S, Garofalo, A M, Wang, H Q, Weisberg, D B, Li,
               Z Y, Jian, X, Eldon, D, Victor, B S, Marinoni, A and
               Hu, Q M, Carvalho, I S, Odstr\v{c}il, T, Wang, L and
               Hyatt, A W, Osborne, T H, Gong, X Z, Qian, J P and
               Huang, J, McClenaghan, J, Holcomb, C T, Hanson, J M* - journal - [DOI](https://doi.org/10.1038/s41586-024-07313-3) | [Nature](https://www.nature.com/articles/s41586-024-07313-3) - The tokamak approach, utilizing a toroidal magnetic field
               configuration to confine a hot plasma, is one of the most
               promising designs for developing reactors that can exploit
               nuclear fusion to generate electrical energy1,2. To reach the
               goal of an economical reactor, most tokamak reactor designs3-10
               simultaneously require reaching a plasma line-averaged density
               above an empirical limit-the so-called Gree... <!-- imported-from-bib -->

**A semi-automated algorithm for designing stellarator divertor and
               limiter plates and application to {HSX}** - **2024** - *Davies, Robert, Feng, Yuhe, Boeyaert, Dieter, Schmitt,
               John C, Gerard, Michael J, Garcia, Kelly A, Schmitz,
               Oliver, Geiger, Benedikt, Henneberg, Sophia A* - journal - [DOI](https://doi.org/10.1088/1741-4326/ad8017) - Abstract We present a semi-automated algorithm for designing
               three-dimensional divertor or limiter plates targeting low heat
               loads. The algorithm designs the plates in two stages: firstly,
               the parallel heat flux distribution is caught on
               vertically-inclined plates at one or several toroidal locations.
               Secondly, the power per unit area is reduced by stretching,
               tilting and bending the plates toroidally. H... <!-- imported-from-bib -->

**Application of Neural Ordinary Differential Equations for
                   tokamak plasma dynamics analysis** - **2024** - *Liu, Zefang, Stacey, Weston M* - journal - [arXiv](https://arxiv.org/abs/2403.01635) - In the quest for controlled thermonuclear fusion, tokamaks
                   present complex challenges in understanding burning plasma
                   dynamics. This study introduces a multi-region
                   multi-timescale transport model, employing Neural Ordinary
                   Differential Equations (Neural ODEs) to simulate the
                   intricate energy transfer processes within tokamaks. Our
                   methodology leverages Neural ODEs for the numeric... <!-- imported-from-bib -->

**Application of interpretable machine learning for
                   cross-diagnostic inference on the {ST40} spherical tokamak** - **2024** - *Pyragius, Tadas, Colgan, Cary, Lowe, Hazel, Janky,
                   Filip, Fontana, Matteo, Cai, Yichen, Naylor, Graham* - journal - [arXiv](https://arxiv.org/abs/2407.18741) - Machine learning models are exceptionally effective in
                   capturing complex non-linear relationships of
                   high-dimensional datasets and making accurate predictions.
                   However, their intrinsic ``black-box'' nature makes it
                   difficult to interpret them or guarantee ``safe behavior''
                   when deployed in high-risk applications such as feedback
                   control, healthcare and finance. This drawback acts ... <!-- imported-from-bib -->

**Artificial intelligence and fusion plasma control: application to
              the {WEST} tokamak** - **2024** - *Kerboua-Benlarbi, Samy* - thesis - [Thesis](https://theses.hal.science/tel-04938923) - Fusion in a magnetically confined plasma is still in the realm of
              fundamental research: in addition to the necessary progress in our
              theoretical knowledge, the operation of current tokamaks remains
              delicate, as it requires substantial human effort each time a new
              experimental scenario is developed. Moreover, the usual
              combination of linear feedback and feedforward control is not very
              robust with respect to the... <!-- imported-from-bib -->

**Avoiding fusion plasma tearing instability with deep
               reinforcement learning** - **2024** - *Seo, Jaemin, Kim, Sangkyeun, Jalalvand, Azarakhsh and
               Conlin, Rory, Rothstein, Andrew, Abbate, Joseph and
               Erickson, Keith, Wai, Josiah, Shousha, Ricardo, Kolemen,
               Egemen* - journal - [DOI](https://doi.org/10.1038/s41586-024-07024-9) - For stable and efficient fusion energy production using a tokamak
               reactor, it is essential to maintain a high-pressure hydrogenic
               plasma without plasma disruption. Therefore, it is necessary to
               actively control the tokamak based on the observed plasma state,
               to manoeuvre high-pressure plasma while avoiding tearing
               instability, the leading cause of disruptions. This presents an
               obstacle-avoidance problem ... <!-- imported-from-bib -->

**Design optimization of nuclear fusion reactor through Deep
                   Reinforcement Learning** - **2024** - *Kim, Jinsu, Seo, Jaemin* - journal - [arXiv](https://arxiv.org/abs/2409.08231) - This research explores the application of Deep Reinforcement
                   Learning (DRL) to optimize the design of a nuclear fusion
                   reactor. DRL can efficiently address the challenging issues
                   attributed to multiple physics and engineering constraints
                   for steady-state operation. The fusion reactor design
                   computation and the optimization code applicable to
                   parallelization with DRL are developed.... <!-- imported-from-bib -->

**Extended database of {2D} {UEDGE} simulations for {KSTAR}
              detachment control with variations of plasma currents** - **2024** - *Zhao, Menglong, Zhu, Ben, Rognlien, Tom, Xu, Xueqiao and
              Meyer, William, Li, Nami, Ma, Xinxing, Kwon, Kyubeen and
              Eldon, David, Lee, Hyungho, Hwang, Junghoo* - journal - [Link](https://ui.adsabs.harvard.edu/abs/2024APS..DPPCM11012/abstract?) - In this work, we extend the database of 50,000 2D UEDGE
              simulations [1] for Machine Learning surrogate models [2] designed
              for KSTAR detachment control. These simulations explore five
              control parameters: core boundary (at PsiN = 0.85) plasma density,
              SOL input power, carbon impurity fraction, perpendicular anomalous
              transport coefficients and the plasma current. Specifically, core
              boundary density ranges from ... <!-- imported-from-bib -->

**Full shot predictions for the {DIII-D} tokamak via deep
                   recurrent networks** - **2024** - *Char, Ian, Chung, Youngseog, Abbate, Joseph and
                   Kolemen, Egemen, Schneider, Jeff* - journal - [arXiv](https://arxiv.org/abs/2404.12416) - Although tokamaks are one of the most promising devices for
                   realizing nuclear fusion as an energy source, there are still
                   key obstacles when it comes to understanding the dynamics of
                   the plasma and controlling it. As such, it is crucial that
                   high quality models are developed to assist in overcoming
                   these obstacles. In this work, we take an entirely data
                   driven approach to learn su... <!-- imported-from-bib -->

**High-fidelity data-driven dynamics model for reinforcement
                   learning-based magnetic control in {HL-3} tokamak** - **2024** - *Wu, Niannian, Yang, Zongyu, Li, Rongpeng, Wei, Ning, Chen, Yihang, Dong, Qianyun, Li, Jiyuan, Zheng,
                   Guohui, Gong, Xinwen, Gao, Feng, Li, Bo, Xu, Min, Zhao, Zhifeng, Zhong, Wulyu* - journal - [arXiv](https://arxiv.org/abs/2409.09238) - The drive to control tokamaks, a prominent technology in
                   nuclear fusion, is essential due to its potential to provide
                   a virtually unlimited source of clean energy. Reinforcement
                   learning (RL) promises improved flexibility to manage the
                   intricate and non-linear dynamics of the plasma encapsulated
                   in a tokamak. However, RL typically requires substantial
                   interaction with a simulator ... <!-- imported-from-bib -->

**Highest fusion performance without harmful edge energy bursts
                   in tokamak** - **2024** - *Kim, Sangkyeun, Shousha, Ricardo, Yang, Seongmoo and
                   Hu, Qiming, Hahn, Sanghee, Jalalvand, Azarakhsh and
                   Park, Jong-Kyu, Logan, Nikolas Christopher, Nelson,
                   Andrew Oakleigh, Na, Yong-Su, Nazikian, Raffi and
                   Wilcox, Robert, Hong, Rongjie, Rhodes, Terry and
                   Paz-Soldan, Carlos, Jeon, Youngmu, Kim, Minwoo, Ko,
                   Wongha, Lee, Jongha, Battey, Alexander, Bortolon,
                   Alessandro, Snipes, Joseph, Kolemen, Egemen* - journal - [arXiv](https://arxiv.org/abs/2405.05452) - The path of tokamak fusion and ITER is maintaining
                   high-performance plasma to produce sufficient fusion power.
                   This effort is hindered by the transient energy burst arising
                   from the instabilities at the boundary of high-confinement
                   plasmas. The application of 3D magnetic perturbations is the
                   method in ITER and possibly in future fusion power plants to
                   suppress this instability and... <!-- imported-from-bib -->

**Implementing deep learning-based disruption prediction in a
               drifting data environment of new tokamak: {HL}-3** - **2024** - *Yang, Zongyu, Zhong, Wulyu, Xia, F, Gao, Zhe, Zhu,
               Xiaobo Xiao, Li, Jiyuan, Hu, Liwen, Xu, Zhaohe, Li,
               Da, Zheng, Guohui, Chen, Y H, Zhang, Junzhao, Li, B, Zhang, Xiaolong, Zhu, Yiren, Tong, Ruihai, Dong, Y B, Zhang, Yipo, Yuan, Boda, Yu, Xin, He, Zongyuhui and
               Tian, Wenjing, Gong, Xinwen, Xu, Min* - journal - [DOI](https://doi.org/10.1088/1741-4326/ada396) - Abstract A deep learning-based disruption prediction algorithm
               has been implemented on a new tokamak, HL-3. An Area Under
               receiver-operator characteristic Curve (AUC) of 0.940 has been
               realized offline over a test campaign involving 72 disruptive and
               240 non-disruptive shots, despite the limited training data
               available from the initial two campaigns. In addition to the
               well-documented challenge of insuff... <!-- imported-from-bib -->

**Learning the dynamics of a one-dimensional plasma model with
               graph neural networks** - **2024** - *Carvalho, Diogo D, Ferreira, Diogo R, Silva, Lu\'{\i}s O* - journal - [DOI](https://doi.org/10.1088/2632-2153/ad4ba6) - Abstract We explore the possibility of fully replacing a plasma
               physics kinetic simulator with a graph neural network-based
               simulator. We focus on this class of surrogate models given the
               similarity between their message-passing update mechanism and the
               traditional physics solver update, and the possibility of
               enforcing known physical priors into the graph construction and
               update. We show that our model ... <!-- imported-from-bib -->

**Leveraging physics-informed neural computing for transport
               simulations of nuclear fusion plasmas** - **2024** - *Seo, J, Kim, I H, Nam, H* - journal - [DOI](https://doi.org/10.1016/j.net.2024.07.048) - For decades, plasma transport simulations in tokamaks have used
               the finite difference method (FDM), a relatively simple scheme to
               solve the transport equations, a coupled set of time-dependent
               partial differential equations. In this FDM approach, typically
               over O(105) time steps are needed for a single discharge, to
               mitigate numerical instabilities induced by stiff transport
               coefficients. It requires sig... <!-- imported-from-bib -->

**Machine learning applications to computational plasma physics
                   and reduced-order plasma modeling: A Perspective** - **2024** - *Faraji, Farbod, Reza, Maryam* - journal - [arXiv](https://arxiv.org/abs/2409.02349) - Machine learning (ML) provides a broad spectrum of tools and
                   architectures that enable the transformation of data from
                   simulations and experiments into useful and explainable
                   science, thereby augmenting domain knowledge. Furthermore,
                   ML-enhanced numerical modelling can revamp scientific
                   computing for real-world complex engineering systems,
                   creating unique opportunities to examine ... <!-- imported-from-bib -->

**Multi-objective Bayesian optimization for design of
               Pareto-optimal current drive profiles in {STEP}** - **2024** - *Brown, Theodore, Marsden, Stephen, Gopakumar, Vignesh and
               Terenin, Alexander, Ge, Hong, Casson, Francis* - journal - [DOI](https://doi.org/10.1109/tps.2024.3382775) <!-- imported-from-bib -->

**Physics-informed deep learning model for line-integral
                   diagnostics across fusion devices** - **2024** - *Wang, Cong, Yang, Weizhe, Wang, Haiping, Yang,
                   Renjie, Li, Jing, Wang, Zhijun, Wei, Yixiong and
                   Huang, Xianli, Hu, Chenshu, Liu, Zhaoyang, Yu,
                   Xinyao, Zou, Changqing, Zhao, Zhifeng* - journal - [arXiv](https://arxiv.org/abs/2412.00087) - Rapid reconstruction of 2D plasma profiles from line-integral
                   measurements is important in nuclear fusion. This paper
                   introduces a physics-informed model architecture called
                   Onion, that can enhance the performance of models and be
                   adapted to various backbone networks. The model under Onion
                   incorporates physical information by a multiplication process
                   and applies the physics-inform... <!-- imported-from-bib -->

**Prediction of plasma rotation velocity and ion temperature
               profiles in {EAST} Tokamak using artificial neural network models** - **2024** - *Lin, Zichao, Zhang, Hongming, Wang, Fudi, Bae, Cheonho, Fu, Jia, Shen, Yongcai, Dai, Shuyu, Jin, Yifei and
               Lu, Dian, Fu, Shengyu, Ji, Huajian, Lyu, Bo* - journal - [DOI](https://doi.org/10.1088/1741-4326/ad73e8) - Abstract Artificial neural network models have been developed to
               predict rotation velocity and ion temperature profiles on the
               EAST tokamak based on spectral measurements from the x-ray
               crystal spectrometer. Both Deep Neural Network (DNN) and
               Convolutional Neural Network (CNN) models have been employed to
               infer line-integrated ion temperatures. The predicted results
               from these two models exhibit a strong... <!-- imported-from-bib -->

**Sample-efficient Bayesian optimisation using known
                   invariances** - **2024** - *Brown, Theodore, Cioba, Alexandru, Bogunovic, Ilija* - journal - [arXiv](https://arxiv.org/abs/2410.16972) - Bayesian optimisation (BO) is a powerful framework for global
                   optimisation of costly functions, using predictions from
                   Gaussian process models (GPs). In this work, we apply BO to
                   functions that exhibit invariance to a known group of
                   transformations. We show that vanilla and constrained BO
                   algorithms are inefficient when optimising such invariant
                   objectives, and provide a method fo... <!-- imported-from-bib -->

**Sample-efficient reinforcement learning with applications in
               nuclear fusion** - **2024** - *Mehta, Viraj* - preprint - [DOI](https://doi.org/10.1184/R1/24944664.V1) - In many practical applications of reinforcement learning (RL), it
               is expensive to observe state transitions from the environment.
               In the problem of plasma control for nuclear fusion, the
               motivating example of this thesis, determining the next state for
               a given state-action pair requires querying an expensive
               transition function which can lead to many hours of computer
               simulation or dollars of scientific ... <!-- imported-from-bib -->

**Sparsified time-dependent Fourier neural operators for fusion
               simulations** - **2024** - *Rahman, Mustafa Mutiur, Bai, Zhe, King, Jacob Robert and
               Sovinec, Carl R, Wei, Xishuo, Williams, Samuel, Liu,
               Yang* - journal - [arXiv](https://arxiv.org/abs/https://pubs.aip.org/aip/pop/article-pdf/doi/10.1063/5.0232503/20281563/123902\_1\_5.0232503.pdf) | [DOI](https://doi.org/10.1063/5.0232503) - This paper presents a sparsified Fourier neural operator for
               coupled time-dependent partial differential equations (ST-FNO) as
               an efficient machine learning surrogate for fluid and
               particle-based fusion codes such as NIMROD (Non-Ideal
               Magnetohydrodynamics with Rotation - Open Discussion) and GTC
               (Gyrokinetic Toroidal Code). ST-FNO leverages the structures in
               the governing equations and utilizes neural op... <!-- imported-from-bib -->

**Time series viewmakers for robust disruption prediction** - **2024** - *Chayapathy, Dhruva, Siebert, Tavis, Spangher, Lucas and
                   Moharir, Akshata Kishore, Patil, Om Manoj, Rea,
                   Cristina* - journal - [arXiv](https://arxiv.org/abs/2410.11065) - Machine Learning guided data augmentation may support the
                   development of technologies in the physical sciences, such as
                   nuclear fusion tokamaks. Here we endeavor to study the
                   problem of detecting disruptions i.e. plasma instabilities
                   that can cause significant damages, impairing the reliability
                   and efficiency required for their real world viability.
                   Machine learning (ML) predictio... <!-- imported-from-bib -->

**Tokamak edge localized mode onset prediction with deep neural
               network and pedestal turbulence** - **2024** - *Joung, Semin, Smith, David R, McKee, G, Yan, Z, Gill,
               K, Zimmerman, J, Geiger, B, Coffee, R, O'Shea, F H, Jalalvand, A, Kolemen, E* - journal - [DOI](https://doi.org/10.1088/1741-4326/ad43fb) - Abstract A neural network, BES-ELMnet, predicting a
               quasi-periodic disruptive eruption of the plasma energy and
               particles known as edge localized mode (ELM) onset is developed
               with observed pedestal turbulence from the beam emission
               spectroscopy system in DIII-D. BES-ELMnet has convolutional and
               fully-connected layers, taking two-dimensional plasma
               fluctuations with a temporal window of size 128 $\mathrm... <!-- imported-from-bib -->

**{Neural-Parareal}: Dynamically training neural operators as
                   coarse solvers for time-parallelisation of fusion {MHD}
                   simulations** - **2024** - *Pamela, S J P, Carey, N, Brandstetter, J, Akers, R, Zanisi, L, Buchanan, J, Gopakumar, V, Hoelzl, M, Huijsmans, G, Pentland, K, James, T, Antonucci,
                   G, {the JOREK Team}* - journal - [arXiv](https://arxiv.org/abs/2405.01355) - The fusion research facility ITER is currently being
                   assembled to demonstrate that fusion can be used for
                   industrial energy production, while several other programmes
                   across the world are also moving forward, such as EU-DEMO,
                   CFETR, SPARC and STEP. The high engineering complexity of a
                   tokamak makes it an extremely challenging device to optimise,
                   and test-based optimisation would b... <!-- imported-from-bib -->

**{TORAX}: A Fast and Differentiable Tokamak Transport
                   Simulator in {JAX}** - **2024** - *Citrin, Jonathan, Goodfellow, Ian, Raju, Akhil and
                   Chen, Jeremy, Degrave, Jonas, Donner, Craig, Felici,
                   Federico, Hamel, Philippe, Huber, Andrea, Nikulin,
                   Dmitry, Pfau, David, Tracey, Brendan, Riedmiller,
                   Martin, Kohli, Pushmeet* - journal - [arXiv](https://arxiv.org/abs/2406.06718) - We present TORAX, a new, open-source, differentiable tokamak
                   core transport simulator implemented in Python using the JAX
                   framework. TORAX solves the coupled equations for ion heat
                   transport, electron heat transport, particle transport, and
                   current diffusion, incorporating modular physics-based and ML
                   models. JAX's just-in-time compilation ensures fast runtimes,
                   while its automati... <!-- imported-from-bib -->

**Automatic identification of edge localized modes in the
               {DIII}-{D} tokamak** - **2023** - *O'Shea, Finn H, Joung, Semin, Smith, David R, Coffee,
               Ryan* - journal - [arXiv](https://arxiv.org/abs/https://pubs.aip.org/aip/aml/article-pdf/doi/10.1063/5.0134001/16820152/026102\_1\_5.0134001.pdf) | [DOI](https://doi.org/10.1063/5.0134001) - Fusion power production in tokamaks uses discharge configurations
               that risk producing strong type I edge localized modes. The
               largest of these modes will likely increase impurities in the
               plasma and potentially damage plasma facing components, such as
               the protective heat and particle divertor. Machine learning-based
               prediction and control may provide for the automatic detection
               and mitigation of these da... <!-- imported-from-bib -->

**Autoregressive transformers for disruption prediction in
                   nuclear fusion plasmas** - **2023** - *Spangher, Lucas, Arnold, William, Spangher, Alexander, Maris, Andrew, Rea, Cristina* - journal - [arXiv](https://arxiv.org/abs/2401.00051) - The physical sciences require models tailored to specific
                   nuances of different dynamics. In this work, we study outcome
                   predictions in nuclear fusion tokamaks, where a major
                   challenge are \textit{disruptions}, or the loss of plasma
                   stability with damaging implications for the tokamak.
                   Although disruptions are difficult to model using physical
                   simulations, machine learning (ML) mod... <!-- imported-from-bib -->

**Bayesian optimization of massive material injection for
               disruption mitigation in tokamaks** - **2023** - *Pusztai, I, Ekmark, I, Bergstr{\"{o}}m, H, Halldestam, P, Jansson, P, Hoppe, M, Vallhagen, O and
               F{\"{u}}l{\"{o}}p, T* - journal - [DOI](https://doi.org/10.1017/s0022377823000193) - A Bayesian optimization framework is used to investigate
               scenarios for disruptions mitigated with combined deuterium and
               neon injection in ITER. The optimization cost function takes into
               account limits on the maximum runaway current, the transported
               fraction of the heat loss and the current quench time. The aim is
               to explore the dependence of the cost function on injected
               densities, and provide insights ... <!-- imported-from-bib -->

**Continuous Convolutional Neural Networks for disruption
                   prediction in nuclear fusion plasmas** - **2023** - *Arnold, William F, Spangher, Lucas, Rea, Christina* - journal - [arXiv](https://arxiv.org/abs/2312.01286) - Grid decarbonization for climate change requires dispatchable
                   carbon-free energy like nuclear fusion. The tokamak concept
                   offers a promising path for fusion, but one of the foremost
                   challenges in implementation is the occurrence of energetic
                   plasma disruptions. In this study, we delve into Machine
                   Learning approaches to predict plasma state outcomes. Our
                   contributions are twofold:... <!-- imported-from-bib -->

**Disruption prediction for future tokamaks using parameter-based
               transfer learning** - **2023** - *Zheng, Wei, Xue, Fengming, Chen, Zhongyong, Chen, Dalong, Guo, Bihao, Shen, Chengshuo, Ai, Xinkun, Wang,
               Nengchao, Zhang, Ming, Ding, Yonghua, Chen, Zhipeng and
               Yang, Zhoujun, Shen, Biao, Xiao, Bingjia, Pan, Yuan* - journal - [DOI](https://doi.org/10.1038/s42005-023-01296-9) - AbstractTokamaks are the most promising way for nuclear fusion
               reactors. Disruption in tokamaks is a violent event that
               terminates a confined plasma and causes unacceptable damage to
               the device. Machine learning models have been widely used to
               predict incoming disruptions. However, future reactors, with much
               higher stored energy, cannot provide enough unmitigated
               disruption data at high performance to tr... <!-- imported-from-bib -->

**Fast equilibrium reconstruction by deep learning on {EAST}
               tokamak** - **2023** - *Lu, Jingjing, Hu, Youjun, Xiang, Nong, Sun, Youwen* - journal - [arXiv](https://arxiv.org/abs/https://pubs.aip.org/aip/adv/article-pdf/doi/10.1063/5.0152318/18032942/075007\_1\_5.0152318.pdf) | [DOI](https://doi.org/10.1063/5.0152318) - A deep neural network is developed and trained on magnetic
               measurements (input) and EFIT poloidal magnetic flux (output) on
               the EAST tokamak. In optimizing the network architecture, we use
               automatic optimization to search for the best hyperparameters,
               which helps in better model generalization. We compare the inner
               magnetic surfaces and last-closed-flux surfaces with those from
               EFIT. We also calculated t... <!-- imported-from-bib -->

**Fourier Neural Operator for Plasma Modelling** - **2023** - *Gopakumar, Vignesh, Pamela, Stanislas, Zanisi, Lorenzo, Li, Zongyi, Anandkumar, Anima, {MAST Team}* - journal - [arXiv](https://arxiv.org/abs/2302.06542) - Predicting plasma evolution within a Tokamak is crucial to
                   building a sustainable fusion reactor. Whether in the
                   simulation space or within the experimental domain, the
                   capability to forecast the spatio-temporal evolution of
                   plasma field variables rapidly and accurately could improve
                   active control methods on current tokamak devices and future
                   fusion reactors. In this work, we dem... <!-- imported-from-bib -->

**Hybridizing physics and neural {ODEs} for predicting plasma
                   inductance dynamics in tokamak fusion reactors** - **2023** - *Wang, Allen M, Garnier, Darren T, Rea, Cristina* - journal - [arXiv](https://arxiv.org/abs/2310.20079) - While fusion reactors known as tokamaks hold promise as a
                   firm energy source, advances in plasma control, and handling
                   of events where control of plasmas is lost, are needed for
                   them to be economical. A significant bottleneck towards
                   applying more advanced control algorithms is the need for
                   better plasma simulation, where both physics-based and
                   data-driven approaches currently fal... <!-- imported-from-bib -->

**Machine learning and Bayesian inference in nuclear fusion
               research: an overview** - **2023** - *Pavone, A, Merlo, A, Kwak, S, Svensson, J* - journal - [DOI](https://doi.org/10.1088/1361-6587/acc60f) - Abstract This article reviews applications of Bayesian inference
               and machine learning (ML) in nuclear fusion research. Current and
               next-generation nuclear fusion experiments require analysis and
               modelling efforts that integrate different models consistently
               and exploit information found across heterogeneous data sources
               in an efficient manner. Model-based Bayesian inference provides a
               framework well suit... <!-- imported-from-bib -->

**Multi-fidelity neural network representation of gyrokinetic
              turbulence** - **2023** - *Neiser, Tom, Meneghini, Orso, Smith, Sterling and
              McClenaghan, Joseph, Slendebroek, Tim, Orozco, David and
              Sammuli, Brian, Staebler, Gary, Hall, Joseph, Belli,
              Emily, Candy, Jeff* - journal - [Link](https://ui.adsabs.harvard.edu/abs/2023APS..DPPPP1039N/abstract) - This presentation will introduce a multi-fidelity neural network
              model of gyrokinetic turbulence GKNN-0, which has been trained and
              validated against a database of 5 million TGLF simulations and
              5000 linear CGYRO simulations with experimental input parameters
              from the DIII-D tokamak. The first half of the presentation will
              review the TGLF saturation rules-SAT0, SAT1, SAT2-and present a
              big data approach to val... <!-- imported-from-bib -->

**Physics-preserving {AI-accelerated} simulations of plasma
                   turbulence** - **2023** - *Greif, Robin, Jenko, Frank, Thuerey, Nils* - journal - [arXiv](https://arxiv.org/abs/2309.16400) - Turbulence in fluids, gases, and plasmas remains an open
                   problem of both practical and fundamental importance. Its
                   irreducible complexity usually cannot be tackled
                   computationally in a brute-force style. Here, we combine
                   Large Eddy Simulation (LES) techniques with Machine Learning
                   (ML) to retain only the largest dynamics explicitly, while
                   small-scale dynamics are described by an M... <!-- imported-from-bib -->

**Towards practical reinforcement learning for tokamak magnetic
                   control** - **2023** - *Tracey, Brendan D, Michi, Andrea, Chervonyi, Yuri and
                   Davies, Ian, Paduraru, Cosmin, Lazic, Nevena and
                   Felici, Federico, Ewalds, Timo, Donner, Craig and
                   Galperti, Cristian, Buchli, Jonas, Neunert, Michael and
                   Huber, Andrea, Evens, Jonathan, Kurylowicz, Paula and
                   Mankowitz, Daniel J, Riedmiller, Martin, {The TCV Team}* - journal - [arXiv](https://arxiv.org/abs/2307.11546) - Reinforcement learning (RL) has shown promising results for
                   real-time control systems, including the domain of plasma
                   magnetic control. However, there are still significant
                   drawbacks compared to traditional feedback control approaches
                   for magnetic confinement. In this work, we address key
                   drawbacks of the RL method; achieving higher control accuracy
                   for desired plasma properties, ... <!-- imported-from-bib -->

**{GS}-{DeepNet}: mastering tokamak plasma equilibria with deep
              neural networks and the Grad-Shafranov equation** - **2023** - *Joung, Semin, Ghim, Y-C, Kim, Jaewook, Kwak, Sehyun and
              Kwon, Daeho, Sung, C, Kim, D, Kim, Hyun-Seok, Bak, J G, Yoon, S W* - journal - [DOI](https://doi.org/10.1038/s41598-023-42991-5) - The force-balanced state of magnetically confined plasmas heated
              up to 100 million degrees Celsius must be sustained long enough to
              achieve a burning-plasma state, such as in the case of ITER, a
              fusion reactor that promises a net energy gain. This force balance
              between the Lorentz force and the pressure gradient force, known
              as a plasma equilibrium, can be theoretically portrayed together
              with Maxwell's equati... <!-- imported-from-bib -->

**{Grad-Shafranov} equilibria via data-free physics informed
                   neural networks** - **2023** - *Jang, Byoungchan, Kaptanoglu, Alan A, Gaur, Rahul and
                   Pan, Shaw, Landreman, Matt, Dorland, William* - journal - [arXiv](https://arxiv.org/abs/2311.13491) - A large number of magnetohydrodynamic (MHD) equilibrium
                   calculations are often required for uncertainty
                   quantification, optimization, and real-time diagnostic
                   information, making MHD equilibrium codes vital to the field
                   of plasma physics. In this paper, we explore a method for
                   solving the Grad-Shafranov equation by using Physics-Informed
                   Neural Networks (PINNs). For PINNs, we opti... <!-- imported-from-bib -->

**{TokaMaker}: An open-source time-dependent Grad-Shafranov
                   tool for the design and modeling of axisymmetric fusion
                   devices** - **2023** - *Hansen, C, Stewart, I G, Burgess, D, Pharr, M and
                   Guizzo, S, Logak, F, Nelson, A O, Paz-Soldan, C* - journal - [arXiv](https://arxiv.org/abs/2311.07719) | [DOI](https://doi.org/10.1016/j.cpc.2024.109111) - In this paper, we present a new static and time-dependent
                   MagnetoHydroDynamic (MHD) equilibrium code, TokaMaker, for
                   axisymmetric configurations of magnetized plasmas, based on
                   the well-known Grad-Shafranov equation. This code utilizes
                   finite element methods on an unstructured triangular grid to
                   enable capturing accurate machine geometry and simple mesh
                   generation from engineering... <!-- imported-from-bib -->

**Building database of {2D} {UEDGE} simulations for the development
               of a surrogate model of divertor detachment control** - **2022** - *Zhao, Menglong, Rognlien, Thomas, Zhu, Ben, Meyer,
               William, Xu, Xueqiao, Bhatia, Harsh, Li, Nami and
               Bremer, Peer-Timo* - journal - [Link](https://ui.adsabs.harvard.edu/abs/2022APS..DPPUP1057Z/abstract) - A large set of 2D UEDGE simulations with currents and cross-field
               drifts based on a generic medium-size tokamak geometry are
               obtained for the development of machine learning surrogate models
               for detachment control. For the current 2D data set, three
               control parameters are varied: gas puff rate, power input and
               impurity fraction. In addition, the values of the perpendicular
               anomalous transport coefficient... <!-- imported-from-bib -->

**Data-driven model for divertor plasma detachment prediction** - **2022** - *Zhu, Ben, Zhao, Menglong, Bhatia, Harsh, Xu,
                   Xue-Qiao, Bremer, Peer-Timo, Meyer, William, Li,
                   Nami, Rognlien, Thomas* - journal - [arXiv](https://arxiv.org/abs/2206.09964) | [DOI](https://doi.org/10.1017/S002237782200085X) | [Link](https://www.cambridge.org/core/product/identifier/S002237782200085X/type/journal_article) - We present a fast and accurate data-driven surrogate model
                   for divertor plasma detachment prediction leveraging the
                   latent feature space concept in machine learning research.
                   Our approach involves constructing and training two neural
                   networks. An autoencoder that finds a proper latent space
                   representation (LSR) of plasma state by compressing the
                   multi-modal diagnostic measurements... <!-- imported-from-bib -->

**Estimation of the electron temperature profile in tokamaks using
               analytical and neural network models** - **2022** - *Morosohk, Shira, Pajares, Andres, Schuster, Eugenio* - conference - [DOI](https://doi.org/10.23919/acc53348.2022.9867844) - Generating energy from nuclear fusion in a tokamak may highly
               benefit from precise control of both kinetic and magnetic
               spatially-varying properties of the plasma (hot ionized gas where
               the fusion reactions take place). The spatial dependence of a
               plasma property, from the core to the edge of the plasma, is
               referred to as profile. Many control algorithms being developed
               require accurate, real-time knowle... <!-- imported-from-bib -->

**Magnetic control of tokamak plasmas through deep reinforcement
                 learning** - **2022** - *Degrave, Jonas, Felici, Federico, Buchli, Jonas and
                 Neunert, Michael, Tracey, Brendan, Carpanese, Francesco, Ewalds, Timo, Hafner, Roland, Abdolmaleki, Abbas and
                 de Las Casas, Diego, Donner, Craig, Fritz, Leslie and
                 Galperti, Cristian, Huber, Andrea, Keeling, James and
                 Tsimpoukelli, Maria, Kay, Jackie, Merle, Antoine and
                 Moret, Jean-Marc, Noury, Seb, Pesamosca, Federico and
                 Pfau, David, Sauter, Olivier, Sommariva, Cristian and
                 Coda, Stefano, Duval, Basil, Fasoli, Ambrogio, Kohli,
                 Pushmeet, Kavukcuoglu, Koray, Hassabis, Demis and
                 Riedmiller, Martin* - journal - [DOI](https://doi.org/10.1038/s41586-021-04301-9) - AbstractNuclear fusion using magnetic confinement, in
                 particular in the tokamak configuration, is a promising path
                 towards sustainable energy. A core challenge is to shape and
                 maintain a high-temperature plasma within the tokamak vessel.
                 This requires high-dimensional, high-frequency, closed-loop
                 control using magnetic actuator coils, further complicated by
                 the diverse requirements across a w... <!-- imported-from-bib -->

**Neural net modeling of equilibria in {NSTX-U}** - **2022** - *Wai, J T, Boyer, M D, Kolemen, E* - journal - [arXiv](https://arxiv.org/abs/2202.13915) - Neural networks (NNs) offer a path towards synthesizing and
                   interpreting data on faster timescales than traditional
                   physics-informed computational models. In this work we
                   develop two neural networks relevant to equilibrium and shape
                   control modeling, which are part of a suite of tools being
                   developed for the National Spherical Torus Experiment-Upgrade
                   (NSTX-U) for fast prediction,... <!-- imported-from-bib -->

**Normalizing flows for likelihood-free inference with fusion
               simulations** - **2022** - *Furia, C S, Churchill, R M* - journal - [DOI](https://doi.org/10.1088/1361-6587/ac828d) - AbstractFluid-based scrape-off layer transport codes, such as
               UEDGE, are heavily utilized in tokamak analysis and design, but
               typically require user-specified anomalous transport coefficients
               to match experiments. Determining the uniqueness of these
               parameters and the uncertainties in them to match experiments can
               provide valuable insights to fusion scientists. We leverage
               recent work in the area of like... <!-- imported-from-bib -->

**Physics-informed machine learning techniques for edge plasma
                   turbulence modelling in computational theory and experiment** - **2022** - *Mathews, Abhilash* - journal - [arXiv](https://arxiv.org/abs/2205.07838) - Edge plasma turbulence is critical to the performance of
                   magnetic confinement fusion devices. Towards better
                   understanding edge turbulence in both theory and experiment,
                   a custom-built physics-informed deep learning framework
                   constrained by partial differential equations is developed to
                   accurately learn turbulent fields consistent with the
                   two-fluid theory from partial observation... <!-- imported-from-bib -->

**Transferable cross-tokamak disruption prediction with deep
                   hybrid neural network feature extractor** - **2022** - *Zheng, Wei, Xue, Fengming, Zhang, Ming, Chen,
                   Zhongyong, Shen, Chengshuo, Ai, Xinkun, Wang,
                   Nengchao, Chen, Dalong, Guo, Bihao, Ding, Yonghua, Chen, Zhipeng, Yang, Zhoujun, Shen, Biao, Xiao,
                   Bingjia, Pan, Yuan* - journal - [arXiv](https://arxiv.org/abs/2208.09594) - Predicting disruptions across different tokamaks is a great
                   obstacle to overcome. Future tokamaks can hardly tolerate
                   disruptions at high performance discharge. Few disruption
                   discharges at high performance can hardly compose an abundant
                   training set, which makes it difficult for current
                   data-driven methods to obtain an acceptable result. A machine
                   learning method capable of trans... <!-- imported-from-bib -->

**Detecting plasma detachment in the Wendelstein 7-{X} stellarator
               using machine learning** - **2021** - *Sz\H{u}cs, M\'{a}t\'{e}, Szepesi, Tam\'{a}s, Biedermann,
               Christoph, Cseh, G\'{a}bor, Jakubowski, Marcin, Kocsis,
               G\'{a}bor, K{\"{o}}nig, Ralf, Krause, Marco, Perseo,
               Valeria, Puig Sitjes, Aleix, {The Team W7-X}* - journal - [DOI](https://doi.org/10.3390/app12010269) - The detachment regime has a high potential to play an important
               role in fusion devices on the road to a fusion power plant.
               Complete power detachment has been observed several times during
               the experimental campaigns of the Wendelstein 7-X (W7-X)
               stellarator. Automatic observation and signaling of such events
               could help scientists to better understand these phenomena. With
               the growing discharge times in f... <!-- imported-from-bib -->

**Neural network surrogate of {QuaLiKiz} using {JET} experimental
               data to populate training space** - **2021** - *Ho, A, Citrin, J, Bourdelle, C, Camenen, Y, Casson, F
               J, van de Plassche, K L, Weisen, H, {JET Contributors}* - journal - [arXiv](https://arxiv.org/abs/https://pubs.aip.org/aip/pop/article-pdf/doi/10.1063/5.0038290/12361366/032305\_1\_online.pdf) | [DOI](https://doi.org/10.1063/5.0038290) - Within integrated tokamak plasma modeling, turbulent transport
               codes are typically the computational bottleneck limiting their
               routine use outside of post-discharge analysis. Neural network
               (NN) surrogates have been used to accelerate these calculations
               while retaining the desired accuracy of the physics-based models.
               This paper extends a previous NN model, known as QLKNN-hyper-10D,
               by incorporating the ... <!-- imported-from-bib -->

**Rapidly-convergent flux-surface shape parameterization** - **2021** - *Arbon, R, Candy, J, Belli, E A* - journal - [DOI](https://doi.org/10.1088/1361-6587/abc63b) | [Link](https://www.osti.gov/servlets/purl/1708848) - Abstract We propose a novel flux-surface parameterization
               suitable for local MHD equilibrium calculations with
               strongly-shaped flux surfaces. The method is based on a
               systematic expansion in a small number of intuitive shape
               parameters, and reduces to the well-known Miller D-shaped
               parameterization in the limit where some of the coefficients are
               set to zero. The new parameterization is valid for up-down
... <!-- imported-from-bib -->

**{N} {EURAL} {DATA} {COMPRESSION} {FOR} {PHYSICS} {PLASMA}
              {SIMULATION}** - **2021** - *Choi, J, Gong, Qian, Pugmire, D, Klasky, S, Churchill,
              M, Ku, S, Lee, Jaemoon, Rangarajan, Anand, Ranka, S* - journal - No link available <!-- imported-from-bib -->

**Fast modeling of turbulent transport in fusion plasmas using
               neural networks** - **2020** - *van de Plassche, K L, Citrin, J, Bourdelle, C, Camenen,
               Y, Casson, F J, Dagnelie, V I, Felici, F, Ho, A and
               Van Mulders, S, {JET Contributors}* - journal - [arXiv](https://arxiv.org/abs/https://pubs.aip.org/aip/pop/article-pdf/doi/10.1063/1.5134126/15772521/022310\_1\_online.pdf) | [DOI](https://doi.org/10.1063/1.5134126) - We present an ultrafast neural network model, QLKNN, which
               predicts core tokamak transport heat and particle fluxes. QLKNN
               is a surrogate model based on a database of 3 \texttimes{} 108
               flux calculations of the quasilinear gyrokinetic transport model,
               QuaLiKiz. The database covers a wide range of realistic tokamak
               core parameters. Physical features such as the existence of a
               critical gradient for the ons... <!-- imported-from-bib -->

**Measuring the electron temperature and identifying plasma
                   detachment using machine learning and spectroscopy** - **2020** - *Samuell, C M, Mclean, A G, Johnson, C A, Glass, F, Jaervinen, A E* - journal - [arXiv](https://arxiv.org/abs/2010.11244) | [DOI](https://doi.org/10.1063/5.0034552) | [AIP](https://pubs.aip.org/rsi/article/92/4/043520/964540/Measuring-the-electron-temperature-and-identifying) - A machine learning approach has been implemented to measure
                   the electron temperature directly from the emission spectra
                   of a tokamak plasma. This approach utilized a neural network
                   (NN) trained on a dataset of 1865 time slices from operation
                   of the DIII-D tokamak using extreme ultraviolet / vacuum
                   ultraviolet (EUV/VUV) emission spectroscopy matched with
                   high-accuracy divertor Thom... <!-- imported-from-bib -->

**Predicting disruptive instabilities in controlled fusion plasmas
               through deep learning** - **2019** - *Kates-Harbeck, Julian, Svyatkovskiy, Alexey, Tang, William* - journal - [DOI](https://doi.org/10.1038/s41586-019-1116-4) - Nuclear fusion power delivered by magnetic-confinement tokamak
               reactors holds the promise of sustainable and clean energy1. The
               avoidance of large-scale plasma instabilities called disruptions
               within these reactors2,3 is one of the most pressing
               challenges4,5, because disruptions can halt power production and
               damage key components. Disruptions are particularly harmful for
               large burning-plasma systems suc... <!-- imported-from-bib -->

**Applications of deep learning to nuclear fusion research** - **2018** - *Ferreira, Diogo R* - journal - [arXiv](https://arxiv.org/abs/1811.00333) - Nuclear fusion is the process that powers the sun, and it is
                   one of the best hopes to achieve a virtually unlimited energy
                   source for the future of humanity. However, reproducing
                   sustainable nuclear fusion reactions here on Earth is a
                   tremendous scientific and technical challenge. Special
                   devices -- called tokamaks -- have been built around the
                   world, with JET (Joint European Toru... <!-- imported-from-bib -->

**Development of a neural network technique for {KSTAR} Thomson
              scattering diagnostics** - **2016** - *Lee, Seung Hun, Lee, J H, Yamada, I, Park, Jae Sun* - journal - [arXiv](https://arxiv.org/abs/https://pubs.aip.org/aip/rsi/article-pdf/doi/10.1063/1.4961079/15616023/11e533\_1\_online.pdf) | [DOI](https://doi.org/10.1063/1.4961079) - Neural networks provide powerful approaches of dealing with
              nonlinear data and have been successfully applied to fusion plasma
              diagnostics and control systems. Controlling tokamak plasmas in
              real time is essential to measure the plasma parameters in situ.
              However, the $\chi$2 method traditionally used in Thomson
              scattering diagnostics hampers real-time measurement due to the
              complexity of the calculations invo... <!-- imported-from-bib -->

**Real-time control of tokamak plasmas: From control of physics to
               physics-based control** - **2011** - *Felici, Federico* - preprint - [DOI](https://doi.org/10.5075/EPFL-THESIS-5203) | [Link](https://infoscience.epfl.ch/handle/20.500.14299/70696) - Stable, high-performance operation of a tokamak requires several
               plasma control problems to be handled simultaneously. Moreover,
               the complex physics which governs the tokamak plasma evolution
               must be studied and understood to make correct choices in
               controller design. In this thesis, the two subjects have been
               merged, using control solutions as experimental tool for physics
               studies, and using physics kno... <!-- imported-from-bib -->

**An advanced disruption predictor for {JET} tested in a simulated
               real-time environment** - **2010** - *Ratt\'{a}, G A, Vega, J, Murari, A, Vagliasindi, G and
               Johnson, M F, de Vries, P C* - journal - [DOI](https://doi.org/10.1088/0029-5515/50/2/025005) - Disruptions are sudden and unavoidable losses of confinement that
               may put at risk the integrity of a tokamak. However, the physical
               phenomena leading to disruptions are very complex and non-linear
               and therefore no satisfactory model has been devised so far
               either for their avoidance or their prediction. For this reason,
               machine learning techniques have been extensively pursued in the
               last years. In this ... <!-- imported-from-bib -->

**Real-time control of a tokamak plasma using neural networks** - **1995** - *Bishop, Chris M, Haynes, Paul S, Smith, Mike E U, Todd,
               Tom N, Trotman, David L* - journal - [DOI](https://doi.org/10.1162/neco.1995.7.1.206) | [Link](https://direct.mit.edu/neco/article/7/1/206-217/5840) - In this paper we present results from the first use of neural
               networks for real-time control of the high-temperature plasma in
               a tokamak fusion experiment. The tokamak is currently the
               principal experimental device for research into the magnetic
               confinement approach to controlled fusion. In an effort to
               improve the energy confinement properties of the high-temperature
               plasma inside tokamaks, recent exper... <!-- imported-from-bib -->

## Implementation Papers

Papers with publicly available code implementations:

- **Physics-informed deep learning model for line-integral diagnostics across fusion devices** - *Wang et al. (2025)* - [Paper](https://doi.org/10.1088/1741-4326/ade0ce) | [Code](https://github.com/calledice/onion) - Neural network model for cross-device line-integral diagnostics with physics constraints

## Tools

### Simulation and Modeling Frameworks

- [TORAX](https://github.com/google-deepmind/torax) - Differentiable tokamak core transport simulator in JAX with ML-surrogate integration (QLKNN neural networks), trajectory optimization, and real-time capable compilation
- [FreeGSNKE](https://github.com/FusionComputingLab/freegsnke) - Free boundary equilibrium solver for tokamaks
- [TokaMaker](https://github.com/OpenFUSIONToolkit/OpenFUSIONToolkit) - An open-source time-dependent Grad-Shafranov tool for the design and modeling of axisymmetric fusion devices ([paper](https://arxiv.org/abs/2311.07719))
- [RAPTOR](https://crppwww.epfl.ch/~sauter/raptor/) - RApid Plasma Transport simulatOR for tokamaks
- [TGLF](https://gafusion.github.io/doc/tglf.html) - Trapped Gyro-Landau Fluid model for tokamak transport
- [FUSE.jl](https://github.com/ProjectTorreyPines/FUSE.jl) - Fusion Simulation Engine in Julia
- [MXHEquilibrium.jl](https://github.com/ProjectTorreyPines/MXHEquilibrium.jl) - MHD equilibrium solver in Julia
- [vmecpp](https://github.com/proximafusion/vmecpp) - C++ implementation of the VMEC stellarator equilibrium code
- [OMFIT](https://gafusion.github.io/OMFIT-source/) - One Modeling Framework for Integrated Tasks with over 110 physics modules, supporting machine learning reduced models and HPC workflow automation. Used by 400+ scientists across 25 institutions
- [OMAS](https://gafusion.github.io/omas/) - Ordered Multi-dimensional Arrays for Magnetic Confinement Fusion, a standardized Python library for storing and manipulating tokamak experimental and simulation data
- [SIMSOPT](https://github.com/hiddenSymmetries/simsopt) - Flexible stellarator optimization framework in Python/C++ with interfaces to VMEC and SPEC, including ML-ready optimization routines and parallelized gradient calculations
- [QuaLiKiz / QLKNN](https://gitlab.com/qualikiz-group) - Quasi-linear gyrokinetic transport model for tokamak plasmas with neural network surrogate (QLKNN) for 10,000x faster predictions
- [Travis Code](https://www.ipp.mpg.de/1060709/travis) - IPP Max Planck Institute plasma physics code


### Machine Learning Frameworks for Plasma Physics

- [FRNN (Fusion Recurrent Neural Network)](https://github.com/PPPLDeepLearning/plasma-python) - Deep learning package for tokamak disruption prediction using recurrent neural networks with stateful LSTM training, multi-machine capabilities, and TensorBoard integration (PPPL)
- [disruption-py](https://github.com/MIT-PSFC/disruption-py) - Physics-based scientific framework for disruption analysis with AI/ML applications supporting multi-tokamak analysis (C-Mod, DIII-D compatibility) (MIT PSFC)

### Data Platforms and Search Tools

- [TokSearch](https://ga-fdp.github.io/toksearch) - Search engine for fusion experimental data ([paper](https://www.sciencedirect.com/science/article/abs/pii/S0920379618301042))
- [DisruptionBench](https://github.com/MIT-PSFC/disruption-benchmark) - First standardized benchmark for tokamak disruption prediction across DIII-D, EAST, and Alcator C-Mod with ~30,000 discharges focusing on model generalizability

## Contributing

Feel free to make a PR or DM me to add something.

To generate README.md with list of papers from bib file use following command: `uv run python add_papers_to_readme.py`
