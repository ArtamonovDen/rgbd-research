# About datasets 

- [About datasets](#about-datasets)
  - [Datasets](#datasets)
  - [Metrics](#metrics)
    - [Evaluation metrics](#evaluation-metrics)
  - [Types of models](#types-of-models)
  - [light field SOD](#light-field-sod)
  - [SOTA](#sota)
  - [RGB SOD](#rgb-sod)

Ещё [куча](https://github.com/taozh2017/RGBD-SODsurvey#related-reviews-and-surveys-to-sod--) стаей с обзорами

Главные сайты:
* http://dpfan.net/d3netbenchmark/
* https://paperswithcode.com/task/rgb-d-salient-object-detection/latest?page=2
* https://github.com/taozh2017/RGBD-SODsurvey

## Datasets

* [RGB-D Salient Object Detection: A Survey](https://arxiv.org/pdf/2008.00230.pdf): часть III. "RGB-D DATASETS" + TABLE VI - овервью, табличка и картинки

* [Rethinking RGB-D Salient Object Detection: Models, Data Sets, and Large-Scale Benchmarks](https://arxiv.org/pdf/1907.06781.pdf): II. "RELATED WORKS" и III. PROPOSED DATASET. Тож самое, но чуваки представили свой датасет [SIP](http://dpfan.net/SIPDataset/)

[Табличка](https://github.com/taozh2017/RGBD-SODsurvey#datasets) со ссыклами на скачивание и основными параметрами

[Сайт с бенчамарками](http://dpfan.net/d3netbenchmark/) от тех же чуваков

Какой-то доп датасет попалася: [ 7-Scenes dataset ](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/)

[RGB-D OD on STERE](https://paperswithcode.com/sota/rgb-d-salient-object-detection-on-stere) на papers with code с бенчмарками и свежими работами.

## Metrics

 [RGB-D Salient Object Detection: A Survey](https://arxiv.org/pdf/2008.00230.pdf): V. MODEL EVALUATION AND ANALYSIS

### Evaluation metrics

* precision-recall (PR)
* F-measure
* mean absolute error (MAE)
* structural measure (S-measure)
* enhanced-alignment measure (E-measure)

Их краткое описание в той же части. Даже формулы есть

## Types of models

Моделей целая куча. Ис список есть в табличках в  [RGB-D Salient Object Detection: A Survey](https://arxiv.org/pdf/2008.00230.pdf) и в табличке на [d3netbenchmark](http://dpfan.net/d3netbenchmark/) и на гитхабе [RGBD-SODsurvey](https://github.com/taozh2017/RGBD-SODsurvey)

Различные виды моделей описаны в [RGB-D Salient Object Detection: A Survey](https://arxiv.org/pdf/2008.00230.pdf) части II.
Они выделяют:

* Traditional/deep models - по способу извлечения признаков,как они пишут
* Fusion-wise models - (fuse - объединять) - различные способы объединения RGB картинки и карты глубины
* Single-stream/multi-stream models - по сути вопрос числа параметов (считай каналов?)
* Attention-aware models - почему бы не засунуть туда аттеншн

## light field SOD

Особняком стоит light field SOD models. Пока непонятно что это, но про это написано в IV части [RGB-D Salient Object Detection: A Survey](https://arxiv.org/pdf/2008.00230.pdf) . Для них даже датасеты другие и отдельная табличка.

> Existing works for SOD can be grouped into three categories according to the input data type, including RGB SOD,
RGB-D SOD, and light field SOD

Для light field SOD используют другие данные - данные светового поля: all-focus image, a focal stack, and a rough depth map
Их тоже разделяют:

* Traditional/Deep Models
* Refinement based Models (refinement - уточнение)

тут уже что-то сложное и с непонятными словами

Соответсвенно ещё одна табличка с датасетами. Пример датасета [LFSD](https://sites.duke.edu/nianyi/publication/saliency-detection-on-light-field/)

Видимо надо отдельно с этим делом разбираться

## SOTA

[paperswithcode](https://paperswithcode.com/task/rgb-d-salient-object-detection/latest?page=2) нам в помощь. Что-то D3Net среди них нет, хотя она вышла в 2019

Обратить внимание на [Most related projects on this website](http://dpfan.net/d3netbenchmark/)

Наверное стоит рассмотреть:

* [BBS-Net](https://arxiv.org/pdf/2007.02713v2.pdf), [github](https://github.com/DengPingFan/BBS-Net), [video](https://drive.google.com/file/d/1qFYkIn7e-Yy3sktFYl6E1xxPXGjBmnuX/view)
* [JL-DCF*](https://arxiv.org/pdf/2008.12134v1.pdf), [github](https://github.com/kerenfu/JLDCF) - но он на caffee, расходимся
* [DASNet](https://arxiv.org/pdf/2006.00269v2.pdf),  [github](https://github.com/JiaweiZhao-git/DASNet) - тут вообще ничё нет
* UCNet-ABP или просто [UCNet](https://paperswithcode.com/paper/uncertainty-inspired-rgb-d-saliency-detection#code) (https://github.com/JingZhang617/UCNet)

## RGB SOD

Для интереса можно посмотреть на https://paperswithcode.com/task/salient-object-detection , без карт глубины. Там тоже есть [UCNet](https://paperswithcode.com/paper/uncertainty-inspired-rgb-d-saliency-detection#code), а ещё [COD](https://paperswithcode.com/paper/cascaded-partial-decoder-for-fast-and#code)