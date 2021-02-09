# Models analysis

- [Models analysis](#models-analysis)
  - [D3Net](#d3net)
    - [Про FLM](#про-flm)
    - [Про DDL](#про-ddl)
    - [Limitations](#limitations)
  - [BBS-NET](#bbs-net)

## D3Net

https://github.com/DengPingFan/D3NetBenchmark

Модель описывают в [части IV](https://arxiv.org/pdf/1907.06781.pdf)

Вот что пишут во введении:

> We propose a simple general model called Deep DepthDepurator Network (D3Net), which learns to automatically discard low-quality depth maps using a novel depth depurator unit (DDU). Thanks to the gate connection mechanism, our D3Net can predict salient objects accurately. Extensive experiments demonstrate that our D3Net  emarkably outperforms prior work on many challenging datasets. Such a general framework design helps to learn

D3Net раздлён на 2 части: FLM(feature learning module) и DDU (depth depurator unit). Определение из словаря: Depurator — One who, or that which, cleanses. [1913 Webster]. FLM состоит ещё из 3 частей, или стримов. В каждый из стримов FLM скармливаются на вход исходные картинки (rgb или карта глубины):

> The FLM (feature learning module) is
utilized to extract the features from different modality

DDU же фильтрует выходы FLM и используется только на тесте.

### Про FLM

FLM состоит из трёх частей: RgbNet, RgbdNet и DepthNet. Все три юзают предобученный vgg16 (наверное чтобы модель не получилось слишком маденькой)и образуют FLM.

Каждая из моделек состоит из трёх одинаковых классов: Decoder, Single Stream и PredLayer:

* PredLayer - самый простой, несколько свёрток с Релу, заканчивается всё сидмоидой
* Decoder - то же самое, но без сигмоиды
* Single_Stream  - сначала VGG, потом ещё пару свёрток сверху

Разница всех трёх моделек только в методе forward в классе MyNet и числе каналов в Single_Stream.

Вход - картинка, выход - карта:
> Specifically, each sub-network receives a re-scaled image I ∈ {Irgb,Irgbd,Idepth} with 224×224 resolution. The goal of FLM is to obtain the corresponding predicted map S ∈ {Srgb,Srgbd,Sdepth}

Ещё есть что-то про апсэмплинг и пробросы через слои, но пока непонятно

### Про DDL

На тесте используется дополнительный gate connection, который как-то отделяет плохие карты глубины.
> The goal of gate connection is to classify depth
maps into reasonable and low-quality ones and not use the
poor ones in the pipeline

А плохие и хорошие карты определяются с помощью гистограммы: на гистограмме хорошей карты видны чёткие пики, а гистограмма плохой - "смазанная":

> “high quality depth maps usually contain clear objects, while
the elements in low-quality depth maps are cluttered"

Сам DDL работает так:

1. На вход подаётся 3 карты глубины:$S \in \{S_{rgb}, S_{rgbd}, S_{depth}\}$
1. Из них надо выбрать самую лучшую (Не особо понятно с размерностью $P$  статье)
    * Сравниваем разницу $S_{rgbd}$ и $S_{depth}$ с порогом $t$.Разница расчитывается с помощью функции $\delta$. Результат сравнения - индекс $F_{cu}\in [0,1]$.

Конечная формула определения оптимальноый карты $P: P=F_{cu} * S_{rgbd} + (1-F_{cu}) * S_{rgb}$. $P$ - saliency map, состоязая из 0 и 1.

Как-то не особо понятно. Как из формулы выше остаются только 0 и 1 в $P$?

### Limitations

Есть у них интеренсая часть про ограничения. Они сами говорят, что ВГГ слишком огромный и даже предлагают заюзать MobileNetv2:
> our simple general framework D3Net consists of
three sub-networks, which may increase the memory on a lightweight device. In a real environment, several strategies can be
considered to avoid this, such as replacing the backbone with
MobileNet V2 , dimension reduction, or using the
recently released ESPNet V2 models

Надо бы посмотреть, наверняка кто-то сделал. Хотя в табличке со всеми моделями в качестве бэкбона максимум резнет. Наверное потому что чуваки охотятся за качеством и циферками побольше. А вот вопрос насколько мобайлнет хуже и как можно уменьшить размер сетки - хороший. Но неплохо бы выбрать нормальную модель, а не д3нет)0)

## BBS-NET

Какая-то большая и сложная.
Статья: https://arxiv.org/pdf/2007.02713v2.pdf

Зато тут есть Ablation Study в пункте 4.3
