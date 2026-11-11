# 便携式清洁机器人视觉系统
# Portable Cleaning Robot Vision

> **来源文件：** `Project-Rob.pdf`，共 15 页。  
> **文档性质：** 课程项目任务书。  

## 目录

- [第 1 页：项目目标、实体系统与时间安排](#P01)
- [第 2—3 页：数据与垃圾类别定义](#P02)
- [第 4—6 页：阶段 2——建立三种标准基线](#P04)
- [第 7 页：阶段 3——研究并测试一项改进](#P07)
- [第 8—9 页：评估要求与可复现预测接口](#P08)
- [第 10—13 页：提交要求与独立核验](#P10)
- [第 14 页：最终课堂活动与考核](#P14)
- [第 15 页：评分标准](#P15)
- [术语对照与阅读提示](#terminology)

---

<a id="P01"></a>
## 第 1 页：项目目标、实体系统与时间安排

<a id="S001"></a>
**来源：** 第 1 页 · S001

**Original:** Artificial Intelligence for Intelligent Engineering. Artificial Intelligence III | Artificial Intelligence for Intelligent Engineering. Portable Cleaning Robot Vision. Robotics Engineering.

**中文:** 面向智能工程的人工智能。人工智能 III｜面向智能工程的人工智能。便携式清洁机器人视觉系统。机器人工程。

### Purpose and final outcome｜目的与最终成果

<a id="S002"></a>
**来源：** 第 1 页 · S002

**Original:** Build the vision module of a small cleaning robot using a portable camera platform. For each newly captured image, locate visible target litter, classify it into the five defined categories, count items in each category, and display PASS if no visible target litter from the five defined categories is present; otherwise, display FAIL. Objects outside these categories are treated as background for the compulsory task.

**中文:** 使用便携式摄像头平台，构建小型清洁机器人的视觉模块。对于每一张新采集的图像，定位其中可见的目标垃圾，将其归入规定的五个类别，并统计各类别的物品数量。如果图像中不存在这五类中的任何可见目标垃圾，则显示 `PASS`；否则显示 `FAIL`。对于本项目的必做任务，五类之外的物体均视为背景。

<a id="S003"></a>
**来源：** 第 1 页 · S003

**Original:** Use a real webcam or connected phone camera and run all image processing, model inference and interface functions on a laptop or desktop. Show boxes, class names, per-object scores, per-class counts and response time. Explain whether a displayed score is a probability or an uncalibrated model score.

**中文:** 使用真实的网络摄像头，或已连接的手机摄像头，并在笔记本电脑或台式电脑上运行全部图像处理、模型推理和界面功能。显示边界框、类别名称、每个物体的得分、各类别的数量以及响应时间。说明所显示的得分是概率，还是未经校准的模型得分。

<a id="S004"></a>
**来源：** 第 1 页 · S004

**Original:** Counts refer to visible items in the current image. Tracking unique objects across video frames is optional. Treat object detection, classification and counting as compulsory final functions.

**中文:** 计数指当前图像中可见物品的数量。跨视频帧追踪唯一物体属于可选功能。目标检测、分类和计数必须作为最终系统的必备功能。

### Physical system｜实体系统

<a id="S005"></a>
**来源：** 第 1 页 · S005

**Original:**
- Build a stable, portable board or frame with a low robot-like camera viewpoint and safe, clean litter props. Wheels, motors and movement control are optional.
- Secure the camera and cables. Allow at least one repeatable adjustment of camera height, distance or tilt and one controlled lighting change. Use fresh camera frames continuously or on demand; stored images alone do not meet the live requirement.
- Navigation, actual litter collection and embedded deployment are optional.

**中文:**
- 搭建稳定、便携的底板或框架，使摄像头采用类似机器人的低位视角，并使用安全、干净的垃圾道具。车轮、电机和运动控制均为可选项。
- 固定摄像头和线缆。摄像头高度、距离或倾角中，至少有一项能够进行可重复的调整，并能够进行一次受控的光照变化。应连续或按需使用摄像头新采集的画面；仅使用已存储的图像不满足实时运行要求。
- 导航、实际收集垃圾以及嵌入式部署均为可选项。

### Stages and dates｜阶段与日期

<a id="T001"></a>
**来源：** 第 1 页 · T001

| Stage／阶段 | Required outcome／必需成果 | Deadline／截止日期 |
|---|---|---|
| 1 | Form a team of 7–8 students and nominate a leader<br>组成 7—8 人的团队，并推选一名组长 | 4 November 2026<br>2026 年 11 月 4 日 |
| 2 | Three standard baselines and initial evaluation<br>三种标准基线及初步评估 | 26 November 2026<br>2026 年 11 月 26 日 |
| 3 | Controlled improvement, final system and individual evidence<br>受控改进、最终系统及个人工作证据 | 13 December 2026<br>2026 年 12 月 13 日 |
| 4 | Final presentation and live demonstration<br>最终汇报与现场演示 | 16 December 2026<br>2026 年 12 月 16 日 |

<a id="S006"></a>
**来源：** 第 1 页 · S006

**Original:** The final session is the only scheduled project class activity; the earlier dates are submission deadlines. All members must attend the final session. Submit team details through the course-announced channel.

**中文:** 最终课堂活动是唯一安排的项目课堂活动；之前列出的日期均为提交截止日期。所有成员必须参加最终课堂活动。通过课程公布的渠道提交团队信息。

---

<a id="P02"></a>
## 第 2 页：数据与垃圾类别定义
## Data and litter definitions｜数据与垃圾类别定义

### Five target categories｜五个目标类别

<a id="T002"></a>
**来源：** 第 2 页 · T002

| Class ID／类别 ID | Category／类别 | Boundary rule／类别边界规则 |
|---|---|---|
| 1 | Cans／罐类 | Metal drink or food cans; exclude detached caps<br>金属饮料罐或食品罐；不包括已脱离的盖子 |
| 2 | Bottles／瓶类 | Plastic or glass bottles; exclude cups and detached caps<br>塑料瓶或玻璃瓶；不包括杯子及已脱离的瓶盖 |
| 3 | Wrappers／包装物 | Flexible packaging bags, packets and film; exclude paper sheets<br>柔性包装袋、小包装袋及包装薄膜；不包括纸张 |
| 4 | Cups／杯类 | Disposable cups of any material; exclude bottles<br>任何材质的一次性杯子；不包括瓶子 |
| 5 | Paper items／纸类物品 | Paper sheets, tissues and cardboard pieces; exclude cups and wrappers<br>纸张、纸巾及纸板碎片；不包括杯子和包装物 |

<a id="S007"></a>
**来源：** 第 2 页 · S007

**Original:** Use mutually exclusive labels. Document how ambiguous examples are resolved before training, using a class catalogue with example photographs. Map each selected public-dataset label to these categories in a saved mapping file. Out-of-scope objects are background; additional classes are optional and must be reported separately.

**中文:** 使用互斥的标签。训练前，应通过附带示例照片的类别目录，记录如何处理存在歧义的样例。将选用的公共数据集标签逐一映射到上述类别，并保存映射文件。范围之外的物体视为背景；额外类别为可选项，且必须单独报告。

### Image labels and visibility｜图像标签与可见性

<a id="S008"></a>
**来源：** 第 2 页 · S008

**Original:** Ground-truth FAIL means at least one verified, non-ignored, in-scope target item is present. Ground-truth PASS means manual inspection confirms no in-scope target item. If ambiguous or ignored regions leave target presence uncertain, flag and report the image separately rather than labelling it PASS. Derive valid image labels and per-class counts consistently from the verified object annotations. A detector producing no boxes does not establish ground truth. The predicted image label is FAIL when at least one retained target detection is present; otherwise it is PASS.

**中文:** 真值标签 `FAIL` 表示图像中至少存在一个已核验、未被忽略且属于任务范围的目标物品。真值标签 `PASS` 表示经人工检查确认不存在任务范围内的目标物品。如果因歧义区域或忽略区域而无法确定目标是否存在，应标记该图像并单独报告，而不是将其标为 `PASS`。有效的图像标签和各类别计数，应以一致方式从已核验的物体标注中推导。检测器没有输出边界框，并不能据此确定真值。预测图像标签的规则是：至少存在一个被保留的目标检测结果时为 `FAIL`；否则为 `PASS`。

<a id="S009"></a>
**来源：** 第 2 页 · S009

**Original:** Define annotation rules for touching items, partly hidden objects, image-edge objects and the minimum visibility needed to identify a class. Save any ignore regions for genuinely ambiguous objects and apply the same evaluation policy to both systems. Count separate visible instances, not fragments of the same item.

**中文:** 制定标注规则，涵盖相互接触的物品、部分被遮挡的物体、位于图像边缘的物体，以及识别类别所需的最低可见程度。对于确实存在歧义的物体，应保存相应的忽略区域，并对两个系统采用相同的评估政策。计数时应统计彼此独立的可见实例，而不是将同一物品的不同可见片段分别计数。

### Sources and annotation｜数据来源与标注

<a id="S010"></a>
**来源：** 第 2 页 · S010

**Original:** You may collect and annotate your own data to prepare the training, validation, and test sets. Some public datasets, such as TACO (https://github.com/pedropro/TACO), may also be suitable. Check the downloaded labels and their meanings carefully. Do not assume that anomaly masks or image-level labels provide complete instance bounding boxes, class labels, and counts for all five required classes. Verify or create the annotations needed for this task. Check each sample’s suitability and explicitly map its object classes to the five-class catalogue.

**中文:** 可以自行采集并标注数据，用于构建训练集、验证集和测试集。一些公共数据集，例如 TACO（https://github.com/pedropro/TACO），也可能适用。应仔细检查下载的标签及其含义。不要假定异常掩膜或图像级标签已经提供全部五个必需类别的完整实例边界框、类别标签和数量。应核验或补充建立本任务所需的标注。检查每个样本是否适用，并明确地将其物体类别映射到五类类别目录。

<a id="S011"></a>
**来源：** 第 2 页 · S011

**Original:** For each image used for object-level training, validation or testing, annotate every visible, in-scope object with a class_id and bounding box (xmin, ymin, xmax, ymax). Use tight, axis-aligned boxes around the visible extent of each object in original-image pixel coordinates, with the origin at the top-left corner; document the box-edge convention. Apply the same documented rules to touching, partly occluded and truncated objects. Record ignored objects or regions and the reasons separately. Derive ground-truth per-class counts from non-ignored object annotations, not from model predictions. Verified images with no target objects must have empty object annotations and zero counts for all five classes. Cases whose target presence or counts cannot be judged must be flagged and reported separately, rather than labelled as empty or PASS.

**中文:** 对每张用于物体级训练、验证或测试的图像，为所有可见且属于任务范围的物体标注 `class_id` 和边界框 `(xmin, ymin, xmax, ymax)`。边界框应与坐标轴平行，并紧密包围每个物体的可见部分；坐标采用原始图像的像素坐标，原点位于左上角，同时记录边界框边缘的坐标约定。对于相互接触、部分遮挡和被图像边界截断的物体，采用同一套已记录的规则。单独记录被忽略的物体或区域及忽略原因。各类别的真值数量应从未被忽略的物体标注中推导，而不能来自模型预测。经核验不存在目标物体的图像，其物体标注必须为空，五类计数必须全部为零。无法判断目标是否存在或无法判断数量的情况，必须标记并单独报告，而不能标为空场景或 `PASS`。

<a id="S012"></a>
**来源：** 第 2—3 页 · S012（原文跨页段落合并）

**Original:** Submit ground-truth annotations separately from prediction files. Include ground_truth_images.csv with image_id, scene_id, split, width, height, annotation_status, the five verified class counts and the image-level label. Include ground_truth_objects.csv with image_id, object_id, class_id, xmin, ymin, xmax, ymax and ignore; object_id must be unique within an image. Document ignored regions separately if used. Every image, including an empty scene, must appear in the image manifest; an empty scene has no object rows. Existing COCO, YOLO or instance-mask annotations may be used if a supplied loader or converter reproduces these fields and counts for evaluation. Check class mapping, image dimensions and converted boxes manually on representative images.

**中文:** 将真值标注与预测文件分开提交。提供 `ground_truth_images.csv`，其中包含 `image_id`、`scene_id`、`split`、`width`、`height`、`annotation_status`、经核验的五类计数及图像级标签。提供 `ground_truth_objects.csv`，其中包含 `image_id`、`object_id`、`class_id`、`xmin`、`ymin`、`xmax`、`ymax` 和 `ignore`；同一图像内的 `object_id` 必须唯一。如果使用忽略区域，应单独记录。每张图像，包括空场景图像，都必须出现在图像清单中；空场景不包含物体记录行。可以使用现有的 COCO、YOLO 或实例掩膜标注，但必须提供加载器或转换器，以便为评估还原上述字段及计数。应选取有代表性的图像，人工检查类别映射、图像尺寸和转换后的边界框。

<a id="P03"></a>
## 第 3 页：独立数据与报告要求

<a id="S013"></a>
**来源：** 第 3 页 · S013

**Original:** Images used only to train or validate image-level classifiers may have image-level labels alone; identify them explicitly. This exception does not apply to the common test images used to evaluate object-capable systems: those images require complete class-and-box ground truth. The same annotated test images can also be used for image-level evaluation.

**中文:** 仅用于训练或验证图像级分类器的图像，可以只有图像级标签，但必须明确指出这些图像。该例外不适用于评估具备物体级输出能力的系统所使用的共同测试图像：这些图像必须具有完整的类别与边界框真值。同一批带标注的测试图像也可以用于图像级评估。

### Independent data and reporting｜独立数据与报告

<a id="S014"></a>
**来源：** 第 3 页 · S014

**Original:** Keep the data units distinct: a scene is one physical arrangement; an image is one captured view; an object instance is one annotated target within an image. A class list or count vector alone is not a complete object-level label. Report scene, image and non-ignored object-instance totals separately, with per-class instance support.

**中文:** 明确区分数据单位：一个场景指一种实体布置；一张图像指一次拍摄得到的视图；一个物体实例指图像内一个已标注的目标。仅有类别列表或计数向量，不构成完整的物体级标签。应分别报告场景总数、图像总数和未被忽略的物体实例总数，并列出各类别的实例样本量。

<a id="S015"></a>
**来源：** 第 3 页 · S015

**Original:** For the initial Stage 2 held-out local evaluation, aim for at least 30 distinct camera scenes, including at least 15 PASS and 15 manually checked FAIL scenes. These scenes are additional to the training and validation data. Count physical scenes, not frames: repeated views, adjacent video frames and lighting variants of the same arrangement belong to one scene group and must remain in the same partition. This is a starting target, not proof that the sample is statistically sufficient. Provide class-and-box ground truth for every image in each object-level test set, include all five classes in the final object-level test and report their instance support; collect more independent scenes where needed.

**中文:** 对于阶段 2 初步的本地留出评估，应以至少 30 个不同的摄像头场景为目标，其中至少包含 15 个 `PASS` 场景和 15 个经人工检查的 `FAIL` 场景。这些场景应独立于训练和验证数据。统计的是实体场景，而不是帧数：同一布置的重复视图、相邻视频帧以及不同光照版本，都属于同一个场景组，必须划入同一个数据分区。这只是起始目标，不能证明样本量在统计上已足够。每个物体级测试集中的每张图像，都应提供类别与边界框真值；最终物体级测试必须包含全部五类，并报告各类别的实例样本量；必要时应额外采集独立场景。

<a id="S016"></a>
**来源：** 第 3 页 · S016

**Original:** Keep related frames, repeated views of the same physical setup and derived lighting variants in the same partition. Maintain training, validation and held-out test lists by scene or physical sample, not by random frame. Respect public-data partitions and report public-data and local-camera results separately. Keep test annotations out of model and threshold selection.

**中文:** 将相关帧、同一实体布置的重复视图，以及由其派生的光照变体，放在同一个数据分区内。按场景或实体样本维护训练、验证和留出测试清单，而不是按帧随机划分。遵循公共数据集原有的数据划分，并分别报告公共数据和本地摄像头数据的结果。不得将测试标注用于模型选择或阈值选择。

---

<a id="P04"></a>
## 第 4 页：阶段 2——建立三种标准基线
## Stage 2 Establish three standard baselines｜阶段 2：建立三种标准基线

<a id="S017"></a>
**来源：** 第 4 页 · S017

**Original:** Due 26 November 2026. Build the physical system and implement THREE meaningfully different established approaches. No novel architecture or research modification is required. Parameter variations of one method do not count as three approaches.

**中文:** 截止日期为 2026 年 11 月 26 日。搭建实体系统，并实现**三种具有实质区别的成熟方法**。不要求提出新架构或进行研究性修改。同一种方法的不同参数设置，不能算作三种方法。

### Baseline coverage｜基线能力覆盖

<a id="S018"></a>
**来源：** 第 4 页 · S018

**Original:** At least ONE of the three baselines must be an object-capable reference system that outputs boxes, the five class labels and per-class counts. The other two may be image-level reference methods. A suitable set is handcrafted features with MLPNN, frozen pretrained CNN features with a linear classifier, and a standard pretrained object detector adapted to the five classes. Other justified established combinations are allowed.

**中文:** 三种基线中，至少有**一种**必须是具备物体级输出能力的参考系统，能够输出边界框、五类标签及各类别计数。另两种可以是图像级参考方法。一种合适的组合是：手工特征加多层感知器神经网络（MLPNN）、冻结的预训练卷积神经网络（CNN）特征加线性分类器，以及适配五个类别的标准预训练目标检测器。也允许采用其他有充分理由的成熟方法组合。

<a id="S019"></a>
**来源：** 第 4 页 · S019

**Original:** The object-capable baseline may use pretrained weights, standard fine-tuning or an established segmentation and classification pipeline. Verify that its class mapping covers this project; a generic checkpoint that lacks the target classes is not sufficient. You do not need to train every model from scratch. Disclose external code, weights, cached features and training provenance.

**中文:** 具备物体级输出能力的基线可以使用预训练权重、标准微调，或成熟的分割与分类流水线。必须核验其类别映射能够覆盖本项目；缺少目标类别的通用检查点不足以满足要求。无需将每个模型都从头训练。应披露外部代码、权重、缓存特征及训练来源。

<a id="S020"></a>
**来源：** 第 4 页 · S020

**Original:** Compare all three baselines on the same image-level PASS/FAIL test set. For the object-capable baseline, derive the image label from its retained detections and additionally evaluate boxes, classes and counts. Do not assign object-level scores to image classifiers that do not produce object outputs.

**中文:** 在同一个图像级 `PASS`/`FAIL` 测试集上比较全部三种基线。对于具备物体级输出能力的基线，应从保留的检测结果中推导图像标签，并额外评估边界框、类别和计数。对于不产生物体级输出的图像分类器，不得赋予物体级评估分数。

### Required implementation and initial evidence｜必需实现与初步证据

<a id="S021"></a>
**来源：** 第 4 页 · S021

**Original:**
- Run all three baselines on newly captured frames, with a model selector. Show the image label and clearly identified model score. The object-capable baseline must also display boxes, classes and counts.
- Use training and validation data to choose parameters, thresholds and preprocessing. State any class balancing and handling of negative images. Record the actual camera, computer, input resolution and dependency versions.
- Evaluate the three baselines fairly at image level. Report object-level metrics for the object-capable reference and save genuine missed objects, false alarms, class confusions and counting errors.
- Before Stage 3, freeze the three checkpoints, preprocessing, class mapping, decision rules and settings. Save a version or checksum and prediction files. Keep the frozen systems runnable.

**中文:**
- 三种基线都应能够处理新采集的画面，并提供模型选择器。显示图像标签，以及明确标明含义的模型得分。具备物体级输出能力的基线还必须显示边界框、类别和计数。
- 使用训练及验证数据选择参数、阈值和预处理方式。说明采用的类别平衡措施，以及如何处理负样本图像。记录实际使用的摄像头、计算机、输入分辨率和依赖版本。
- 在图像级公平评估三种基线。对具备物体级输出能力的参考系统，报告物体级指标，并保存真实出现的漏检物体、误报、类别混淆和计数错误。
- 在阶段 3 开始之前，冻结三个模型检查点、预处理方式、类别映射、决策规则和设置。保存版本号或校验和，以及预测文件。确保冻结后的系统仍可运行。

### Stage 2 submission｜阶段 2 提交要求

<a id="S022"></a>
**来源：** 第 4 页 · S022

**Original:** **Stage 2 Group Report:** The report should follow the provided template, format.docx, and must not exceed 10 pages. The separately submitted decision_stage2.md file does not count towards the report page limits.

**中文:** **阶段 2 小组报告：** 报告应使用提供的模板 `format.docx`，且**不得超过 10 页**。单独提交的 `decision_stage2.md` 文件不计入报告页数限制。

<a id="T003"></a>
**来源：** 第 4—6 页 · T003（报告章节要求表，按原表行序分列于三页）

| Section title／章节标题 | Required content — Original | 必需内容——中文 |
|---|---|---|
| 1. Engineering Task and Physical Setup<br>1. 工程任务与实体装置 | Describe the task, required outputs and operating conditions. Include a photograph or labelled diagram of the rig. State the camera, lighting, computer, processing device, resolution and operating restrictions. | 描述任务、必需输出及运行条件。附上装置照片或带标注的示意图。说明摄像头、光照、计算机、处理设备、分辨率及运行限制。 |

<a id="P05"></a>
## 第 5 页：阶段 2 报告要求（续）

<a id="T004"></a>
**来源：** 第 5 页 · T004（接 T003）

| Section title／章节标题 | Required content — Original | 必需内容——中文 |
|---|---|---|
| 2. Classes, Data Sources and Annotations<br>2. 类别、数据来源与标注 | Present the five-class catalogue with IDs, names and examples. Explain sources, label mapping, exclusions and annotation checks. Define rules for touching, occluded, truncated and ambiguous objects, and manually verify negative scenes. Define PASS/FAIL and visible-object counts. Document per-object class IDs and bounding boxes, coordinate conventions, ignored annotations and count derivation. Identify the ground-truth files and any annotation converter. | 给出五类类别目录，包括 ID、名称和示例。说明数据来源、标签映射、排除项及标注检查。制定相互接触、被遮挡、被截断以及有歧义物体的处理规则，并人工核验负场景。定义 `PASS`/`FAIL` 及可见物体计数。记录各物体的类别 ID 与边界框、坐标约定、被忽略的标注及计数推导方式。明确指出真值文件及所用标注转换器。 |
| 3. Data Partitions and Evaluation Protocol<br>3. 数据划分与评估协议 | Report training, validation and test support by image, independent scene and class. Keep related frames and derived images in the same partition. Identify split files, parameter-selection procedures and freezing points. Separate public-data and local-camera results and disclose data dependence. Report non-ignored object-instance support separately from image and scene totals, and identify which partitions have complete class-and-box annotations. | 按图像、独立场景和类别报告训练、验证及测试的样本量。相关帧及派生图像必须位于同一数据分区。明确划分文件、参数选择程序和冻结节点。分别报告公共数据与本地摄像头数据的结果，并披露数据之间的依赖关系。未被忽略的物体实例样本量，应与图像总数、场景总数分开报告；说明哪些数据分区具有完整的类别与边界框标注。 |
| 4. Three Baseline Methods and Implementation<br>4. 三种基线方法与实现 | Explain each method, selection rationale, pipeline and supported outputs. Identify external code, pretrained weights, frozen and trained components, actual training settings, validation selection and final configurations. Report blockers and incomplete functions. Include a comparison table; parameter variations alone do not constitute distinct methods. | 说明每种方法、选择理由、处理流水线及支持的输出。明确外部代码、预训练权重、冻结和训练的组件、实际训练设置、验证选择过程及最终配置。报告阻碍因素和未完成功能。附上比较表；仅有参数差异不能构成不同的方法。 |
| 5. Baseline Results and Response Time<br>5. 基线结果与响应时间 | Compare all three methods on the same supported image-level task and held-out cases. Report confusion matrices, precision, recall, F1 and support. Separately report detection and counting metrics for object-capable methods. Report median and 95th-percentile live latency from at least 20 fresh frames per compared system, with hardware and timing boundaries. Retain failed inputs and report coverage. | 在同一个三种方法均支持的图像级任务及同一批留出样例上比较三种方法。报告混淆矩阵、精确率、召回率、F1 及样本量。对具备物体级输出能力的方法，单独报告检测及计数指标。对每个参与比较的系统，使用至少 20 个新采集帧，报告实时延迟的中位数和第 95 百分位数，并说明硬件与计时起止边界。保留处理失败的输入，并报告覆盖情况。 |
| 6. Weaknesses and Failure Analysis<br>6. 弱点与失败分析 | For each baseline, distinguish inherent limitations from observed problems. Cite image/run IDs, predictions and ground truth. Explain possible causes, an alternative explanation and further checks. Discuss engineering consequences. Summarise using: **Baseline / Method limitation / Observed problem and evidence / Possible cause / Engineering consequence**. | 针对每种基线，区分方法固有的局限与实际观察到的问题。引用图像／运行 ID、预测结果及真值。解释可能原因、一种替代解释，以及进一步核查方法。讨论工程后果。按以下格式汇总：**基线／方法局限／观察到的问题及证据／可能原因／工程后果**。 |
| 7. Initial Decision Record and Improvement Plan<br>7. 初始决策记录与改进计划 | Include one compact record with: **decision ID; initial claim and reason; alternative considered; planned verification and fixed conditions; evidence references; observed baseline result; retained or revised conclusion; limitation; evidence that would change the decision**. Select one evidenced weakness and specify the proposed improvement, hypothesis, primary metric, expected cost and independent final test plan. Save this before implementing the improvement. | 提供一条简明记录，包含：**决策 ID；初始主张及理由；考虑过的替代方案；计划中的验证方式与固定条件；证据引用；观察到的基线结果；保留或修订后的结论；局限；会使决策改变的证据**。选择一个有证据支持的弱点，并明确拟议改进、假设、主要指标、预期代价及独立最终测试计划。在实施改进之前保存该记录。 |

<a id="P06"></a>
## 第 6 页：阶段 2 报告与提交材料（续）

<a id="T005"></a>
**来源：** 第 6 页 · T005（接 T004）

| Section title／章节标题 | Required content — Original | 必需内容——中文 |
|---|---|---|
| 8. Advice, Reproducibility and Completion Status<br>8. 建议、可复现性与完成状态 | For one material suggestion, record its source—**LLM, instructor, peer, paper or self**—and a short relevant extract where applicable. Record **accept/modify/reject/defer**, reason, implementation status and a code/configuration/output reference. State no LLM use honestly. List reproducibility files and unresolved work. Keep this evidence in the existing report or review appendix without duplication. | 对一条有实质影响的建议，记录其来源——**大语言模型、教师、同学、论文或本人**——并在适用时附上简短的相关摘录。记录**接受／修改／拒绝／暂缓**的决定、理由、实施状态，以及代码／配置／输出的引用。如未使用大语言模型，应如实说明。列出可复现性文件与尚未解决的工作。将这些证据保留在现有报告或评审附录中，避免重复。 |
| 9. LLM Use<br>9. 大语言模型使用情况 | Briefly state which LLM tools were used and how they supported this stage, such as planning, literature understanding, coding, debugging or result interpretation. Give one concrete example and explain how the output was checked, modified or rejected. Distinguish LLM assistance from the team’s own work. If no LLM was used, state this explicitly. Cross-reference existing advice records rather than duplicating them. | 简要说明使用了哪些大语言模型工具，以及它们如何支持本阶段，例如规划、文献理解、编码、调试或结果解释。给出一个具体例子，说明如何检查、修改或拒绝模型输出。区分大语言模型提供的辅助与团队自身的工作。若未使用大语言模型，应明确说明。交叉引用已有建议记录，而不要重复记录。 |

<a id="S023"></a>
**来源：** 第 6 页 · S023

**Original:** **Short video:** Submit an MP4 video shorter than 2 minutes. Demonstrate fresh camera input, switching between all three baselines, positive and negative examples, and the object-capable output. Include a multi-item litter scene and a manually checked scene with no target litter.

**中文:** **短视频：** 提交一个时长**不足 2 分钟**的 MP4 视频。演示摄像头新采集输入、三种基线之间的切换、正负样例，以及物体级输出。视频必须包含一个有多件垃圾的场景，以及一个经人工核验不含目标垃圾的场景。

<a id="S024"></a>
**来源：** 第 6 页 · S024

**Original:** **Reproducibility package:** source code, requirements, three frozen model files, configuration files, class mapping, split lists, evaluation code, ground-truth image manifests and per-object class-and-box annotations for the reported group test, plus annotations and loaders/converters used for project training and validation and prediction files. Include the batch interface specified later.

**中文:** **可复现材料包：** 包括源代码、依赖要求、三个冻结模型文件、配置文件、类别映射、数据划分清单、评估代码、所报告小组测试的真值图像清单及逐物体类别与边界框标注，以及项目训练和验证所用的标注与加载器／转换器、预测文件。还应提供后文规定的批量处理接口。

<a id="S025"></a>
**来源：** 第 6 页 · S025

**Original:** **Stage 2 decision record:** Submit review/decision_stage2.md (maximum one page) at Stage 2, before implementing the proposed improvements. Include the baseline findings, proposed changes and reasons, planned experiments, evaluation criteria, and expected trade-offs. Assign a decision ID to each proposed change so that its outcome can be traced in the final evaluation. Submit through the course-announced submission channel. Use the subject or package name AI Project Stage 2 Report Group XX. Large videos may use an accessible cloud link. The instructor will independently check the submitted frozen models after the deadline.

**中文:** **阶段 2 决策记录：** 在阶段 2 提交 `review/decision_stage2.md`，**最多一页**，且应在实施拟议改进之前提交。记录基线发现、拟议修改及理由、计划实验、评估标准和预期权衡。为每项拟议修改分配一个决策 ID，以便在最终评估中追踪其结果。通过课程公布的提交渠道提交。邮件主题或材料包名称使用 `AI Project Stage 2 Report Group XX`。较大的视频可以提供可访问的云端链接。截止日期之后，教师将独立检查所提交的冻结模型。

---

<a id="P07"></a>
## 第 7 页：阶段 3——研究并测试一项改进
## Stage 3 Investigate and test one improvement｜阶段 3：研究并测试一项改进

<a id="S026"></a>
**来源：** 第 7 页 · S026

**Original:** Due 13 December 2026. Use a measured baseline weakness to define a research question, implement at least ONE justified change and evaluate its benefit and cost. Completing the compulsory object functions alone does not establish a research contribution or an improvement.

**中文:** 截止日期为 2026 年 12 月 13 日。根据已测得的基线弱点，确定一个研究问题，实施至少**一项有充分依据的修改**，并评估其收益与代价。仅仅完成必需的物体级功能，并不能证明产生了研究贡献或实现了改进。

### Define the question and freeze the reference｜确定问题并冻结参考系统

<a id="S027"></a>
**来源：** 第 7 页 · S027

**Original:** Example: Small wrappers are missed at a low camera viewpoint. Will image tiling improve object recall, and what happens to false positives and latency?

**中文:** 示例：摄像头处于低位视角时，小型包装物容易漏检。图像切块是否能提高物体召回率？误报和延迟又会发生怎样的变化？

<a id="S028"></a>
**来源：** 第 7 页 · S028

**Original:** Use development and validation evidence to choose the question. Record the frozen reference version, hypothesis, proposed change, primary metric, expected cost, selection rule and controlled conditions before the final comparison. For a claim about localisation, classes or counts, use an object-capable reference with the same output definitions. Image classifiers support only image-level comparisons.

**中文:** 使用开发和验证阶段的证据来选择研究问题。在最终比较之前，记录冻结的参考版本、假设、拟议修改、主要指标、预期代价、选择规则及受控条件。若要提出有关定位、类别或计数的主张，必须使用具有相同输出定义且具备物体级输出能力的参考系统。图像分类器只能支持图像级比较。

<a id="S029"></a>
**来源：** 第 7 页 · S029

**Original:** Use an object-capable baseline as the primary reference for the final system improvement. The final pipeline must retain all compulsory detection, classification and counting functions. Image-level classifier comparisons may be reported as supplementary evidence.

**中文:** 最终系统改进应以具备物体级输出能力的基线作为主要参考。最终流水线必须保留全部必需的检测、分类和计数功能。图像级分类器的比较可以作为补充证据报告。

### Connect research to the implemented change｜将研究与实际修改建立联系

<a id="S030"></a>
**来源：** 第 7 页 · S030

**Original:** Use at least one relevant research paper or technical study. Explain the original problem or method, which idea you adapt to this project, where it appears in your code or acquisition procedure, and what result would support or challenge the adaptation. A citation alone is insufficient. An instructor-discussed research problem may be used, but distinguish your classroom adaptation from the original method.

**中文:** 至少使用一篇相关研究论文或技术研究资料。解释其原始问题或方法、你将其中哪个想法适配到本项目、这一想法具体体现在代码或采集流程的何处，以及什么结果会支持或质疑该适配。仅列出引用并不够。可以使用教师讨论过的研究问题，但必须区分你们在课堂项目中的适配与原始方法。

<a id="S031"></a>
**来源：** 第 7 页 · S031

**Original:** Possible changes include targeted data augmentation, local image regions, a justified preprocessing step, a different established representation, score calibration, staged verification or runtime optimisation. Merely downloading a larger model is insufficient without a reasoned hypothesis and fair measurement. New neural-network architectures are not required.

**中文:** 可考虑的修改包括有针对性的数据增强、使用局部图像区域、有依据的预处理步骤、采用另一种成熟的表示方式、得分校准、分阶段核验或运行时优化。如果缺乏合理假设和公平测量，仅下载一个更大的模型不足以满足要求。不要求设计新的神经网络架构。

### Run a controlled comparison｜进行受控比较

<a id="S032"></a>
**来源：** 第 7 页 · S032

**Original:**
- Use the same previously untouched Stage 3 group test scenes, image labels, ground-truth classes and bounding boxes, matching rules and metrics for the frozen reference and improved pipeline. Tune only on training/validation data.
- If Stage 2 test results influenced the Stage 3 design, treat them as exploratory and keep a new independent Stage 3 test set untouched until the final comparison. Do not repeatedly test candidate settings on the final test set.
- Evaluate at least TWO difficult conditions, such as small objects, shadows, low light, partial occlusion, clutter or changed camera height/angle. Record the condition and sample support. Keep paired views or derived versions within the same held-out scene group. Include normal-condition results for both systems to check whether the change harms performance in ordinary scenes.
- Change one main factor at a time where possible. If several changes are combined, report a pipeline comparison; use an ablation before attributing the gain to one component. Report failed trials and increased misses, false alarms, delay or resource use.

**中文:**
- 冻结的参考系统与改进后的流水线，应使用同一批此前从未用于开发或调参的阶段 3 小组测试场景，以及相同的图像标签、类别与边界框真值、匹配规则和指标。仅在训练／验证数据上调参。
- 如果阶段 2 的测试结果影响了阶段 3 的设计，应将这些结果视为探索性结果，并保留一个新的、独立的阶段 3 测试集，在最终比较之前不予使用。不得在最终测试集上反复测试候选设置。
- 至少评估**两种困难条件**，例如小物体、阴影、低照度、部分遮挡、杂乱背景或摄像头高度／角度变化。记录条件及样本量。成对视图或派生版本应保留在同一个留出场景组内。两个系统都应报告正常条件下的结果，以检查该修改是否损害普通场景中的性能。
- 尽可能每次只改变一个主要因素。如果组合了多项修改，应报告整个流水线的比较；在将性能提升归因于某一组件之前，应进行消融实验。报告失败尝试，以及增加的漏检、误报、延迟或资源消耗。

### Integrate and preserve evidence｜集成系统并保存证据

<a id="S033"></a>
**来源：** 第 7 页 · S033

**Original:** Run the final complete system on fresh camera images using the same rig. Keep the selected frozen reference and improved system selectable without code edits. Preserve the original plan, revisions, implementation versions, configurations, raw predictions and calculated results. A rigorous negative result can earn strong marks; do not hide it.

**中文:** 使用同一套装置，让最终完整系统处理摄像头新采集的图像。所选的冻结参考系统与改进系统，应能在无需修改代码的情况下切换。保存原始计划、修订记录、实现版本、配置、原始预测及计算结果。严谨的否定性结果同样可以获得高分；不要隐瞒。

---

<a id="P08"></a>
## 第 8 页：评估要求
## Evaluation requirements｜评估要求

<a id="S034"></a>
**来源：** 第 8 页 · S034

**Original:** Report training/validation evidence, group-held-out results and instructor-held results separately. High image accuracy alone does not establish reliable localisation or counting. Include sample sizes and per-class support; acknowledge uncertainty from small or dependent samples.

**中文:** 分别报告训练／验证证据、小组留出测试结果和教师保留测试集结果。仅有较高的图像准确率，并不能证明定位或计数可靠。报告样本量及各类别样本量；承认样本较少或样本间存在依赖时带来的不确定性。

<a id="T006"></a>
**来源：** 第 8 页 · T006

| Capability／能力 | Required measures and rules — Original | 必需指标与规则——中文 |
|---|---|---|
| Image decisions<br>图像判定 | Confusion matrix TP, FP, FN, TN; precision, recall and F1 for FAIL; support for both image classes. Balanced accuracy may be added. | 报告混淆矩阵中的 TP、FP、FN、TN；以 `FAIL` 为正类的精确率、召回率和 F1；两个图像类别各自的样本量。可额外报告平衡准确率。 |
| Object classes and boxes<br>物体类别与边界框 | Report per-class precision and recall and test support. Match predictions to non-ignored ground-truth objects of the same class one-to-one at IoU ≥ 0.50. Each object and detection may be matched once. Unmatched retained predictions are FP; unmatched labelled targets are FN. Ground truth must include class IDs and bounding boxes for all evaluable targets; image labels or count vectors alone are insufficient. Document how predictions overlapping ignored annotations are handled and apply the same rule to both systems. | 报告各类别的精确率、召回率及测试样本量。在 IoU ≥ 0.50 的条件下，将预测与同类别、未被忽略的真值物体进行一对一匹配。每个物体和每个检测结果只能匹配一次。未匹配的保留预测为 FP；未匹配的已标注目标为 FN。所有可评估目标的真值都必须包含类别 ID 和边界框；仅有图像标签或计数向量不够。记录与忽略标注重叠的预测如何处理，并对两个系统采用相同规则。 |
| Counting<br>计数 | For each class, report mean absolute error across test images, including zero-count cases. Also report the proportion of images with the entire five-class count vector correct. State whether counts use the same retained detections as the displayed result. Derive ground-truth counts from the same non-ignored object annotations used for box evaluation. Exclude and report images with unjudgeable target counts rather than assigning zero counts. | 对每个类别，报告在全部测试图像上的平均绝对误差，包括该类数量为零的情况。另报告五类完整计数向量全部正确的图像比例。说明计数是否使用与界面显示结果相同的保留检测。真值数量应从用于边界框评估的同一批未被忽略的物体标注中推导。目标数量无法判定的图像，应排除并报告，而不是将其数量设为零。 |
| Critical errors<br>关键错误 | Report missed target objects, false detections and image-level false alarms separately. Show class confusions and examples of incorrect counts. | 分别报告目标物体漏检、错误检测以及图像级误报。展示类别混淆和计数错误示例。 |
| Controlled improvement<br>受控改进 | Give reference and improved values, absolute differences, difficult-condition support and trade-offs. Use identical test inputs and rules. Do not compare incomparable outputs or different test sets as evidence of a gain. | 给出参考系统与改进系统的指标值、绝对差值、困难条件下的样本量及相关权衡。使用完全相同的测试输入和规则。不得将不可比的输出或不同测试集的结果作比较，以此作为提升证据。 |
| Live response<br>实时响应 | Use at least 20 fresh frames per compared system. Measure from acquired frame availability to completion of processing and the display update call. Include preprocessing, inference and postprocessing; synchronise GPU work if used. Report median and 95th-percentile latency. | 每个参与比较的系统至少使用 20 个新采集帧。从采集帧可供程序使用的时刻开始计时，直到处理完成并完成显示更新调用。计入预处理、推理和后处理；如使用 GPU，应同步 GPU 工作。报告延迟中位数和第 95 百分位数。 |
| Resource use<br>资源使用 | Report CPU/GPU, image resolution, model-file size, batching and inference settings; report memory if measured. Do not infer sustained camera FPS solely from reciprocal single-frame latency. | 报告 CPU／GPU、图像分辨率、模型文件大小、批处理及推理设置；若测量了内存占用，也应报告。不能仅根据单帧延迟的倒数推断摄像头的持续帧率（FPS）。 |

### Scores and matching｜得分与匹配

<a id="S035"></a>
**来源：** 第 8 页 · S035

**Original:** Choose detection thresholds and any non-maximum suppression settings on validation data and freeze them. Use the same evaluator for both systems. Save the thresholds and the IoU matching implementation. Optional mAP results must state the IoU range and evaluator; they supplement the required measures. Match predictions in descending score order to the highest-IoU unmatched eligible ground-truth object; resolve ties consistently in the evaluator.

**中文:** 在验证数据上选择检测阈值及所有非极大值抑制设置，并将其冻结。两个系统使用同一个评估器。保存阈值及 IoU 匹配实现。可选的 mAP 结果必须说明 IoU 范围和评估器；它们只是对必需指标的补充。按预测得分从高到低进行匹配，将每个预测匹配到符合条件、尚未匹配且 IoU 最高的真值物体；在评估器中以一致方式处理并列情况。

<a id="S036"></a>
**来源：** 第 8 页 · S036

**Original:** Report the score type and which outcome a higher score supports. If a rule-based image decision has no image-level confidence, record its score as null rather than inventing one; retain the object scores.

**中文:** 报告得分类型，以及更高的得分支持哪一种结果。如果基于规则的图像判定没有图像级置信度，应将其得分记为 `null`，而不是虚构一个数值；同时保留物体得分。

<a id="P09"></a>
## 第 9 页：可复现预测接口

<a id="S037"></a>
**来源：** 第 9 页 · S037（接第 8 页评估要求）

**Original:** Include both manually checked negative scenes and multi-item positive scenes. A good image-level F1 score can coexist with poor item recall; keep those conclusions separate.

**中文:** 必须同时包含经人工检查的负场景，以及含多件物品的正场景。较好的图像级 F1 分数可能与较差的物体召回率同时存在；应分别得出这两方面的结论。

## Reproducible prediction interface｜可复现预测接口

<a id="S038"></a>
**来源：** 第 9 页 · S038

**Original:** Provide one documented batch command that loads each submitted model by ID without editing code. The command below defines the required arguments; your implementation must support equivalent behaviour. Define the supported model IDs in your README using baseline1, baseline2, baseline3, and final, and state which method and saved model each ID loads. Include an environment dependency file (requirements.txt or environment.yml) and setup instructions in your submission. Test the setup and command in a clean environment before submission.

**中文:** 提供一个有文档说明的批量命令，使其能够通过 ID 加载每个提交的模型，而无需修改代码。下面的命令规定了必需参数；实现必须支持等效行为。在 README 中使用 `baseline1`、`baseline2`、`baseline3` 和 `final` 定义支持的模型 ID，并说明每个 ID 加载的方法和已保存模型。提交材料中应包含环境依赖文件（`requirements.txt` 或 `environment.yml`）及安装配置说明。提交前，应在干净环境中测试安装配置过程及该命令。

<a id="S039"></a>
**来源：** 第 9 页 · S039（原文命令，英文与中文共用）

```bash
python predict.py --input_dir TEST_FOLDER --model MODEL_ID --output_dir RESULTS
```

**Original:** Replace MODEL_ID with one of the model IDs defined above.

**中文:** 将 `MODEL_ID` 替换为上文定义的一个模型 ID。

### Input contract｜输入约定

<a id="S040"></a>
**来源：** 第 9 页 · S040

**Original:** Use image filenames as image_id values and document supported formats. Include model IDs, dependency versions, model files and configuration files. The supplied class catalogue is fixed for an evaluation. Inference must not read ground-truth labels or train on the test input.

**中文:** 将图像文件名用作 `image_id`，并记录支持的图像格式。提供模型 ID、依赖版本、模型文件和配置文件。一次评估过程中，所提供的类别目录保持固定。推理不得读取真值标签，也不得使用测试输入进行训练。

<a id="S041"></a>
**来源：** 第 9 页 · S041

**Original:** The five class IDs are defined in this sheet. Empty target scenes must still appear in the image-level and count outputs with zero counts.

**中文:** 五个类别 ID 由本任务书规定。不含目标的空场景仍必须出现在图像级输出和计数输出中，并将各类别数量记为零。

### Required output files｜必需输出文件

<a id="T007"></a>
**来源：** 第 9 页 · T007

| File／文件 | Required contents — Original | 必需内容——中文 |
|---|---|---|
| `images.csv` | One row per image: image_id, model_id, status, predicted_label (PASS or FAIL), prediction_score, score_type. | 每张图像一行：`image_id`、`model_id`、`status`、`predicted_label`（`PASS` 或 `FAIL`）、`prediction_score`、`score_type`。 |
| `objects.csv` | For object-capable systems: image_id, model_id, detection_id, class_id, xmin, ymin, xmax, ymax, score. Use original-image pixel coordinates with a top-left origin. An empty scene has no object rows. This file contains predictions, not ground-truth labels; keep it separate from ground_truth_objects.csv and use the same coordinate convention. | 对于具备物体级输出能力的系统：`image_id`、`model_id`、`detection_id`、`class_id`、`xmin`、`ymin`、`xmax`、`ymax`、`score`。使用原始图像的像素坐标，以左上角为原点。空场景不含物体记录行。此文件包含预测结果，而非真值标签；应与 `ground_truth_objects.csv` 分开保存，并采用相同的坐标约定。 |
| `counts.csv` | One row per image and class: image_id, class_id, observed_count. Include all five classes, including zero counts. | 每张图像的每个类别各一行：`image_id`、`class_id`、`observed_count`。必须包含全部五类，包括数量为零的类别。 |
| `run_config.json` | Model/version, class catalogue, thresholds, preprocessing, coordinate convention, score meanings and hardware. Include output-capability flags so image-only models are not mistaken for object detectors. | 记录模型／版本、类别目录、阈值、预处理、坐标约定、得分含义及硬件。加入输出能力标志，避免将仅支持图像级输出的模型误认为目标检测器。 |

<a id="S042"></a>
**来源：** 第 9 页 · S042

**Original:** Image-only baselines are exempt from objects.csv and counts.csv. They must not fabricate object outputs. The compulsory object-capable baseline and final improved system must provide all output files. Use a documented non-success status for failed inputs; do not silently omit them.

**中文:** 仅支持图像级输出的基线，无须提供 `objects.csv` 和 `counts.csv`，也不得虚构物体级输出。必需的物体级基线与最终改进系统，必须提供全部输出文件。对于处理失败的输入，应使用有文档说明的非成功状态，不得悄然将其省略。

### Evaluation package｜评估材料包

<a id="S043"></a>
**来源：** 第 9 页 · S043

**Original:** Include the evaluator, ground-truth image manifests, class-and-box annotations, ignore rules and annotation loaders/converters, split IDs and raw predictions behind every reported metric. Keep offline batch timing separate from live camera latency. Predictions should be reproducible with the submitted frozen versions and settings.

**中文:** 提供评估器、真值图像清单、类别与边界框标注、忽略规则及标注加载器／转换器、数据划分 ID，以及每项已报告指标所依据的原始预测。将离线批处理计时与实时摄像头延迟分开报告。使用提交的冻结版本及设置，应能复现预测结果。

---

<a id="P10"></a>
## 第 10 页：提交要求与独立核验
## Submissions and independent verification｜提交要求与独立核验

### Stage 3 group submission｜阶段 3 小组提交

<a id="S044"></a>
**来源：** 第 10 页 · S044

**Original:** Submit by 13 December 2026 through the course-announced channel. Use AI Project Stage 3 Material Group XX as the subject or package name. Include the following:

**中文:** 在 2026 年 12 月 13 日之前，通过课程公布的渠道提交。邮件主题或材料包名称使用 `AI Project Stage 3 Material Group XX`。应包含以下材料：

<a id="S045"></a>
**来源：** 第 10 页 · S045

**Original:** **Stage 3 Group Report:** The report should follow the provided template, format.docx, and must not exceed 15 pages. The separately submitted decision_final.md file does not count towards the report page limits.

**中文:** **阶段 3 小组报告：** 报告应使用提供的模板 `format.docx`，且**不得超过 15 页**。单独提交的 `decision_final.md` 文件不计入报告页数限制。

<a id="T008"></a>
**来源：** 第 10 页 · T008

| Section title／章节标题 | Required content — Original | 必需内容——中文 |
|---|---|---|
| 1. Engineering Purpose and Final System<br>1. 工程目的与最终系统 | Describe the final system, operating scope and required outputs. Show the rig and explain changes since the baseline stage. | 描述最终系统、运行范围及必需输出。展示装置，并解释相较于基线阶段进行了哪些修改。 |
| 2. Frozen Baselines and Chosen Reference<br>2. 冻结基线与所选参考系统 | Summarise the three unchanged baselines and identify their versions. Explain the chosen reference using its capabilities, measured weaknesses and suitability for the proposed comparison. Reference earlier details rather than repeating them. | 概述三个保持不变的基线，并注明各自版本。从系统能力、测得的弱点及其对拟议比较的适用性出发，解释为何选择该参考系统。引用之前的详细说明，避免重复。 |
| 3. Data, Annotations and Independent Test Protocol<br>3. 数据、标注与独立测试协议 | Explain sources, class mapping, annotation checks and partitions. Report independent scene, class and condition support. Describe how final test data remained unused during development. Identify earlier test results used for exploratory decisions. Confirm complete class-and-box ground truth for every object-level test image; report non-ignored object-instance support and annotation versions. | 说明数据来源、类别映射、标注检查及数据划分。报告独立场景、类别和各条件下的样本量。说明如何确保最终测试数据在开发期间未被使用。指出哪些早期测试结果被用于探索性决策。确认每张物体级测试图像均具有完整的类别与边界框真值；报告未被忽略的物体实例样本量及标注版本。 |
| 4. Research Source and Adaptation<br>4. 研究来源与适配 | Discuss at least one relevant paper or technical study. Explain the idea adopted, its relevance, what was modified or omitted, and where it appears in the implementation. A citation alone is insufficient. | 讨论至少一篇相关论文或技术研究。解释所采用的想法、其相关性、修改或省略了哪些部分，以及该想法体现在实现的何处。仅提供引用并不够。 |
| 5. Hypothesis and Decision Record Revisions<br>5. 假设与决策记录修订 | Revisit the original decision ID, weakness, hypothesis, alternatives, primary metric, planned verification, fixed conditions and expected costs. Preserve the initial record. Explain any revisions, when they occurred and why. Do not rewrite the original plan to match the final outcome. | 回顾最初的决策 ID、弱点、假设、替代方案、主要指标、计划验证、固定条件及预期代价。保留初始记录。说明所有修订、发生时间及原因。不得为迎合最终结果而改写原始计划。 |
| 6. Actual Implementation and Advice Audit<br>6. 实际实现与建议审查 | Explain changed and unchanged components, settings and versions. Link planned changes to actual code/configuration and run IDs. For one material suggestion, identify its source, relevant extract, **accept/modify/reject/defer** decision, reason and implementation status. Include unsuccessful or unimplemented suggestions where relevant; cross-reference existing records instead of duplicating them. | 说明修改与未修改的组件、设置及版本。将计划中的修改与实际代码／配置及运行 ID 对应起来。针对一条有实质影响的建议，指出其来源、相关摘录、**接受／修改／拒绝／暂缓**决定、理由及实施状态。必要时包含未成功或尚未实施的建议；交叉引用已有记录，不要重复记录。 |
| 7. Controlled Results and Difficult Conditions<br>7. 受控实验结果与困难条件 | Compare reference and improved systems on identical inputs, labels and evaluation rules. Report values and absolute differences under normal conditions and **at least two difficult conditions**, with support. Separate image-level, object-level and counting results. Use ablations before attributing a combined improvement to one component. | 在完全相同的输入、标签及评估规则下，比较参考系统与改进系统。报告正常条件及**至少两种困难条件**下的指标值、绝对差值与样本量。分别报告图像级、物体级和计数结果。将组合改进归因于某一组件之前，应先进行消融实验。 |

<a id="P11"></a>
## 第 11 页：阶段 3 报告要求与个人提交（续）

<a id="T009"></a>
**来源：** 第 11 页 · T009（接 T008）

| Section title／章节标题 | Required content — Original | 必需内容——中文 |
|---|---|---|
| 8. Failures and Alternative Explanations<br>8. 失败情况与替代解释 | Present remaining errors, new errors and unsuccessful trials using actual evidence references. Explain plausible causes and alternatives. State whether the targeted weakness was reduced, unchanged or worsened, and what further evidence would distinguish explanations. | 使用真实证据引用展示仍然存在的错误、新出现的错误和未成功的尝试。解释合理的原因与替代解释。说明所针对的弱点是减轻、没有变化还是恶化，以及需要哪些进一步证据来区分不同解释。 |
| 9. Live Performance and Resource Trade-offs<br>9. 实时性能与资源权衡 | Demonstrate both systems on fresh camera input. Report median and 95th-percentile latency from at least 20 frames per system, hardware, resolution, timing boundaries and model-file size; include memory if measured. Explain whether benefits justify additional delay or resource use. | 使用摄像头新采集输入演示两个系统。每个系统至少使用 20 帧，报告延迟中位数和第 95 百分位数、硬件、分辨率、计时边界及模型文件大小；若测量了内存占用，也应报告。解释所获收益是否足以抵偿增加的延迟或资源消耗。 |
| 10. Final Decision, Limitations and Next Action<br>10. 最终决策、局限与下一步行动 | State whether the original recommendation is retained or revised and why. Distinguish implemented changes from demonstrated benefits. Explain limitations, recommend one justified next engineering action, and specify conditions or new evidence that would change the decision. A rigorous negative result is acceptable. | 说明保留还是修订原始建议，以及原因。区分已实施的修改与已经证实的收益。解释局限，提出一项有依据的后续工程行动，并明确哪些条件或新证据会改变当前决策。严谨的否定性结果可以接受。 |
| 11. Reproducibility and Contributions<br>11. 可复现性与成员贡献 | List frozen baselines, final implementation, configurations, split files, annotations, raw predictions, evaluator and commands. Link conclusions to reproducible outputs. Summarise member contributions and reference separate individual reports. Identify incomplete work and separate instructor-held results from group evaluation. Include ground-truth image manifests, per-object class-and-box annotations, ignore definitions and any loader/converter. | 列出冻结基线、最终实现、配置、数据划分文件、标注、原始预测、评估器及命令。将结论对应到可复现的输出。概述成员贡献，并引用单独的个人报告。指出未完成的工作，并将教师保留测试集结果与小组评估分开。提供真值图像清单、逐物体类别与边界框标注、忽略定义及所有加载器／转换器。 |
| 12. LLM Use<br>12. 大语言模型使用情况 | Briefly state which LLM tools were used and how they supported this stage, such as planning, literature understanding, coding, debugging or result interpretation. Give one concrete example and explain how the output was checked, modified or rejected. Distinguish LLM assistance from the team’s own work. If no LLM was used, state this explicitly. Cross-reference existing advice records rather than duplicating them. | 简要说明使用了哪些大语言模型工具，以及它们如何支持本阶段，例如规划、文献理解、编码、调试或结果解释。给出一个具体例子，说明如何检查、修改或拒绝模型输出。区分大语言模型的辅助与团队自身的工作。若未使用大语言模型，应明确说明。交叉引用已有建议记录，而不要重复记录。 |

<a id="S046"></a>
**来源：** 第 11 页 · S046

**Original:** **Reproducibility files:** source code, requirements, THREE unchanged baseline files, final improved model, versions/configurations, class mapping, split lists, evaluator, ground-truth image manifests and per-object class-and-box annotations for group tests, plus annotations and loaders/converters used for project training and validation, raw prediction files and run instructions. Include rig photographs and a backup video.

**中文:** **可复现文件：** 包括源代码、依赖要求、**三个保持不变的基线文件**、最终改进模型、版本／配置、类别映射、数据划分清单、评估器、小组测试的真值图像清单及逐物体类别与边界框标注，以及项目训练和验证所用的标注与加载器／转换器、原始预测文件及运行说明。附上装置照片和备用视频。

<a id="S047"></a>
**来源：** 第 11 页 · S047

**Original:** **Final decision record:** Submit review/decision_final.md (maximum one page) with the final submission. For each Stage 2 decision ID, state what was implemented, revised, or rejected and why; identify the implementation version, experiment/run IDs, and result locations; and evaluate the outcome against the original criteria. Include unsuccessful changes, remaining limitations, and reasons for selecting the final implementation. Preserve the original Stage 2 record and document any later changes to the plan.

**中文:** **最终决策记录：** 最终提交时一并提交 `review/decision_final.md`，**最多一页**。针对阶段 2 的每个决策 ID，说明实施、修订或拒绝了哪些内容及原因；指出实现版本、实验／运行 ID 以及结果位置；按照原始标准评估结果。包含未成功的修改、仍然存在的局限，以及选择最终实现方案的理由。保留阶段 2 的原始记录，并记录此后对计划所作的任何变更。

### Individual submission｜个人提交

<a id="S048"></a>
**来源：** 第 11 页 · S048

**Original:** By 13 December 2026, each student submits a report independently using AI Project Student ID Individual Report Group XX. The report should follow the provided template, format.docx, and must not exceed 2 pages.

**中文:** 每位学生须在 2026 年 12 月 13 日之前独立提交一份报告，提交名称使用 `AI Project Student ID Individual Report Group XX`。报告应使用提供的模板 `format.docx`，且**不得超过 2 页**。

<a id="P12"></a>
## 第 12 页：个人报告要求与教师核验

<a id="T010"></a>
**来源：** 第 12 页 · T010

| Section title／章节标题 | Required content — Original | 必需内容——中文 |
|---|---|---|
| 1. My Specific Contribution<br>1. 我的具体贡献 | Identify tasks personally completed and distinguish individual from shared work. Cite specific code, annotations, experiments, figures or logs. Avoid unsupported statements such as “I helped with programming.” | 明确本人完成的任务，区分个人工作与共同工作。引用具体代码、标注、实验、图表或日志。避免“我参与了编程”这类缺乏证据支持的表述。 |
| 2. One Technical Decision — D<br>2. 一项技术决策——D | Identify the decision ID, problem, initial judgement, alternative considered and reasons for the chosen method. If advice influenced the decision, state its source and why it was accepted, modified, rejected or deferred. | 指出决策 ID、问题、初始判断、考虑过的替代方案，以及选择该方法的理由。如果某条建议影响了决策，说明其来源，以及接受、修改、拒绝或暂缓该建议的原因。 |
| 3. One Personally Checked Result — E<br>3. 一项亲自核验的结果——E | Explain one result you personally verified, the checking procedure and what the evidence supports. Cite an image/run ID, prediction row, figure or log and identify relevant controlled conditions. | 解释一项由本人亲自核验的结果、核验程序，以及证据支持什么结论。引用图像／运行 ID、预测记录行、图表或日志，并指出相关受控条件。 |
| 4. Limitation and Alternative Explanation — E/T<br>4. 局限与替代解释——E/T | Explain one limitation and one plausible alternative explanation for the observed result. State what additional check would help distinguish the explanations. | 针对观察到的结果，解释一项局限和一种合理的替代解释。说明增加哪项核查有助于区分这些解释。 |
| 5. Next Engineering Action — T<br>5. 下一步工程行动——T | Recommend one concrete action justified by the evidence. Explain the expected benefit, cost or risk, conditions or missing evidence, and what finding would change your recommendation. | 提出一项由证据支持的具体行动。解释预期收益、代价或风险、适用条件或缺失的证据，以及什么发现会改变你的建议。 |

<a id="S049"></a>
**来源：** 第 12 页 · S049

**Original:** Use the same reasoning dimensions as Lab 2 and Lab 3: D—problem definition and method selection; E—evidence design and interpretation; T—engineering trade-offs, limitations and transfer.

**中文:** 使用与实验 2、实验 3 相同的推理维度：D——问题定义与方法选择；E——证据设计与解释；T——工程权衡、局限与迁移应用。

<a id="S050"></a>
**来源：** 第 12 页 · S050

**Original:** Sections 2–5 analyse one selected decision, including one personally checked result, a plausible alternative explanation and a justified next action. Shared code or data do not justify identical personal explanations.

**中文:** 第 2—5 节应分析同一个选定决策，包括一项亲自核验的结果、一种合理的替代解释，以及一项有依据的后续行动。共享代码或数据，并不能成为个人解释完全相同的理由。

<a id="S051"></a>
**来源：** 第 12 页 · S051

**Original:** Submit confidential, evidence-based contribution feedback separately through the course-announced channel. Describe your own and teammates’ contributions using specific tasks and evidence. Do not include this feedback in the shared group submission.

**中文:** 通过课程公布的渠道，单独提交保密的、基于证据的贡献反馈。使用具体任务及证据描述自己和队友的贡献。不要将该反馈放入共享的小组提交材料。

### Instructor checks at both stages｜两个阶段的教师核验

<a id="S052"></a>
**来源：** 第 12 页 · S052

**Original:** After Stage 2, the instructor runs all three frozen models on a hidden Stage 2 set. Image-level baselines receive image-level checks; the object-capable baseline also receives class, box and count checks. Submitted code and group split records may also be used to reproduce the group-reported results. General feedback may be given without releasing hidden labels or images.

**中文:** 阶段 2 结束后，教师将在隐藏的阶段 2 数据集上运行全部三个冻结模型。图像级基线接受图像级核验；具备物体级输出能力的基线还将接受类别、边界框和计数核验。教师也可能使用提交的代码与小组数据划分记录，复现小组报告中的结果。可以在不公开隐藏标签或图像的情况下提供总体反馈。

<a id="S053"></a>
**来源：** 第 12 页 · S053

**Original:** After Stage 3, the instructor uses a NEW hidden set, distinct from Stage 2. The unchanged relevant reference and improved system receive identical images and evaluation rules. Compare only supported outputs. Hidden inputs and labels must not be used for training, setting selection or repeated trial-and-error submissions. The final live demonstration is an additional check.

**中文:** 阶段 3 结束后，教师将使用一个与阶段 2 不同的**全新隐藏数据集**。相关的、保持不变的参考系统与改进系统，将使用相同的图像及评估规则。只比较各系统支持的输出。隐藏输入及标签不得用于训练、设置选择或反复试错式提交。最终现场演示是额外的核验。

<a id="S054"></a>
**来源：** 第 12 页 · S054

**Original:** The instructor checks image labels, object class IDs, bounding boxes and derived counts manually. Report instructor-supplied scores separately and add them after verification. A lower score on a new environment does not itself prove fabrication. Material discrepancies lead to review of code, splits, labels, thresholds and records before any academic-integrity conclusion.

**中文:** 教师会人工检查图像标签、物体类别 ID、边界框及由此推导的计数。教师提供的分数应单独报告，并在核验后补入。在新环境中得分较低，本身并不能证明存在造假。若出现重大差异，应先审查代码、数据划分、标签、阈值及记录，再作出任何学术诚信方面的结论。

<a id="P13"></a>
## 第 13 页：大语言模型辅助与负责任实践
### LLM assistance and responsible practice｜大语言模型辅助与负责任实践

<a id="S055"></a>
**来源：** 第 13 页 · S055

**Original:** LLMs may assist with reading, planning, coding and troubleshooting. Preserve the relevant prompt and response or a concise extract for a material suggestion; state what you accepted, changed or rejected, why, and which code/run/result checks it. If you use no LLM, record that honestly. No credit depends on accepting an LLM suggestion. A plan written by an LLM is not evidence that the experiment was executed.

**中文:** 大语言模型可以辅助阅读、规划、编码和故障排查。对于有实质影响的建议，应保留相关提示词及回复，或其简明摘录；说明接受、修改或拒绝了哪些内容、原因是什么，以及通过哪些代码／运行／结果对其进行了核验。如果未使用大语言模型，应如实记录。是否接受大语言模型的建议，不影响获得分数。大语言模型编写的计划，不是实验已实际执行的证据。

<a id="S056"></a>
**来源：** 第 13 页 · S056

**Original:** Use safe sample components and secure the inspection rig. Respect image permissions, acknowledge papers, datasets, external code and weights, and avoid identifiable people or private data. Keep genuine logs; never fabricate predictions, measurements or contributions.

**中文:** 使用安全的样品与组件，并牢固固定检测装置。遵守图像使用权限要求，注明论文、数据集、外部代码和权重的来源，避免使用包含可识别个人身份的人物影像或私人数据。保留真实日志；绝不伪造预测、测量结果或贡献。

<a id="S057"></a>
**来源：** 第 13 页 · S057

**Original:** Do not submit complete private chats or images containing identifiable personal information. Relevant advice may be documented using a short extract.

**中文:** 不要提交完整的私人聊天记录，也不要提交包含可识别个人身份信息的图像。相关建议可使用简短摘录进行记录。

---

<a id="P14"></a>
## 第 14 页：最终课堂活动与考核
## Final session and assessment｜最终课堂活动与考核

<a id="S058"></a>
**来源：** 第 14 页 · S058

**Original:** 16 December 2026. All members attend and bring the portable rig, camera, computer, power supply, model files and a backup video. Submit the printed Stage 3 Group Report before the presentation. A video is a backup, not a substitute for fresh camera input.

**中文:** 时间为 2026 年 12 月 16 日。所有成员必须到场，并携带便携式装置、摄像头、计算机、电源、模型文件及备用视频。在汇报开始前提交打印版阶段 3 小组报告。视频只是备用材料，不能代替摄像头新采集输入。

### Group slot｜小组时段安排

<a id="T011"></a>
**来源：** 第 14 页 · T011

| Activity／活动 | Time／时间 | Required evidence／必需证据 |
|---|---|---|
| Presentation／汇报 | 5 minutes／5 分钟 | Purpose, rig, baseline comparison, research change, strongest result and limitation. Keep the live demonstration separate.<br>说明目的、装置、基线比较、研究性修改、最有说服力的结果及局限。现场演示应单独进行。 |
| Questions／问答 | 4 minutes／4 分钟 | Any member may be asked to explain data validity, model behaviour, a personal contribution or an experimental decision.<br>任何成员都可能被要求解释数据有效性、模型行为、个人贡献或某项实验决策。 |
| Live demonstration／现场演示 | 8 minutes, including setup／8 分钟，含设备设置 | Show the submitted reference and final system; fresh positive and checked-negative scenes; multiple items, predicted classes, boxes and counts.<br>展示所提交的参考系统与最终系统；使用新采集的正场景及经核验的负场景；展示多个物品、预测类别、边界框和计数。 |

---

<a id="P15"></a>
## 第 15 页：评分标准
## Assessment rubric｜评分标准

<a id="T012"></a>
**来源：** 第 15 页 · T012（将原页的评分与证据列恢复为同一张表）

| Criterion／评分项 | Marks／分值 | Evidence — Original | 证据——中文 |
|---|---|---|---|
| Stage 2 basic AI system<br>阶段 2 基础人工智能系统 | 15 | Three distinguishable frozen baselines; at least one complete object-capable reference; portable live system; instructor-checkable outputs. | 三种可区分的冻结基线；至少一个完整的物体级参考系统；便携式实时系统；可供教师核验的输出。 |
| Stage 2 report and video<br>阶段 2 报告与视频 | 10 | Valid class definitions, annotations and splits; comparable image-level results; object metrics where supported; reproducible video and files. | 有效的类别定义、标注及数据划分；可比较的图像级结果；在系统支持时提供物体级指标；可复现的视频及文件。 |
| Stage 3 research improvement<br>阶段 3 研究性改进 | 15 | Measured weakness, testable hypothesis, implemented change and a clear link from research idea to implementation. | 经测量确认的弱点、可检验的假设、已实施的修改，以及从研究想法到具体实现的明确联系。 |
| Stage 3 experiments and report<br>阶段 3 实验与报告 | 10 | Untouched comparison, two difficult conditions, class/box/count evidence, errors and latency/resource trade-offs; separate instructor results. | 使用此前未使用的测试数据进行比较；两种困难条件；类别／边界框／计数证据；错误以及延迟／资源权衡；单独报告教师测试结果。 |
| Final live demonstration<br>最终现场演示 | 15 | Fresh camera input, five-class localisation/counting, image decision and runnable reference/improved versions. | 摄像头新采集输入、五类定位／计数、图像判定，以及可运行的参考／改进版本。 |
| Presentation and questions<br>汇报与问答 | 15 | Clear explanation, warranted conclusions, limitations and technically grounded answers. | 清晰的解释、有充分依据的结论、对局限的说明，以及具有技术依据的回答。 |
| Adherence to instructions<br>遵守要求 | 5 | Complete usable submissions, honest provenance, safe operation and deadlines. | 完整且可用的提交材料、如实披露来源、安全操作及遵守截止日期。 |
| Individual report<br>个人报告 | 7.5 | Specific personal work, a defensible decision and personally verified evidence. | 具体的个人工作、经得起论证的决策，以及亲自核验的证据。 |
| Contribution feedback<br>贡献反馈 | 7.5 | Evidence-based peer feedback moderated by the instructor using demonstrable contributions. | 基于证据的同伴反馈，由教师根据可证明的贡献进行审核调整。 |
| Total／总分 | 100 | Group assessment 85 marks; individual assessment 15 marks. | 小组考核 85 分；个人考核 15 分。 |

---

<a id="terminology"></a>
## 术语对照

> 本节为翻译附录，不属于原文新增的项目要求。

| English | 本文译法 |
|---|---|
| baseline | 基线 |
| reference system | 参考系统 |
| object-capable system | 具备物体级输出能力的系统 |
| image-level / object-level | 图像级／物体级 |
| ground truth | 真值 |
| bounding box | 边界框 |
| scene / image / object instance | 场景／图像／物体实例 |
| support | 样本量；涉及物体时指实例样本量 |
| held-out test set | 留出测试集 |
| frozen checkpoint | 冻结的模型检查点 |
| retained detection | 被保留的检测结果 |
| ignore region / ignored annotation | 忽略区域／被忽略的标注 |
| precision / recall | 精确率／召回率 |
| false positive (FP) / false negative (FN) | 假阳性（误报）／假阴性（漏检） |
| true positive (TP) / true negative (TN) | 真阳性／真阴性 |
| intersection over union (IoU) | 交并比 |
| mean absolute error | 平均绝对误差 |
| mean average precision (mAP) | 平均精度均值 |
| non-maximum suppression | 非极大值抑制 |
| score calibration | 得分校准 |
| ablation | 消融实验 |
| latency / 95th-percentile latency | 延迟／延迟的第 95 百分位数 |
| provenance | 来源及形成过程 |
| rigorous negative result | 严谨的否定性结果，即未证实预期收益的研究结果 |

## 阅读提示与排版说明

- `PASS`／`FAIL` 保留原样；判定含义以正文定义为准：无可见目标垃圾为 `PASS`，存在可见目标垃圾为 `FAIL`。
- 原文是课程项目任务书，不是研究论文；没有需要提取的插图或独立陈列公式。所有表格均已完整重排，跨页表格按行恢复。
- 页码指原 PDF 页码；S 编号标记正文块，T 编号标记表格。第 2—3 页跨页段落合并于 S012。重复的课程页眉、项目名称页脚及页脚页码未逐页重译。
- 原文提交名称中的 `XX`、`Student ID` 以及命令中的大写参数保留为原文占位内容。
- 未省略实质性段落、提交要求或评分项；不对原文未规定的技术方案、数量或规则作补充推断。
