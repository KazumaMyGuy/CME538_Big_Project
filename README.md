# CME538_Big_Project
#2022 Kaggle DS & ML Survey
#Methodology:
#● Survey data collection
#  ○ The full list of questions that were asked (along with the provided answer choices) can be
    found in the file: kaggle_survey_2022_answer_choices.pdf. The file contains footnotes
    that describe exactly which questions were asked to which respondents. Respondents
    with the most experience were asked the most questions. For example, students and
    unemployed persons were not asked questions about their employer. Likewise,
    respondents that do not write code were not asked questions about writing code. For
    more detail, see the footnotes associated with each question in the document
    kaggle_survey_2022_answer_choices.pdf.
#  ○ The survey was live from 08/16/2022 to 09/16/2022. We allowed respondents to
  complete the survey at any time during that window. An invitation to participate in the
  survey was sent to the entire Kaggle community (anyone opted-in to the Kaggle Email
  List) via email. The survey was also promoted on the Kaggle website via popup "nudges"
  and on the Kaggle Twitter channel.
#● Survey data processing
#  ○ The survey responses can be found in the file kaggle_survey_2022_responses.csv.
  Responses to multiple choice questions (only a single choice can be selected) were
  recorded in individual columns. Responses to multiple selection questions (multiple
  choices can be selected) were split into multiple columns (with one column per answer
  choice). The data released under a CC 2.0 license:
  https://creativecommons.org/licenses/by/2.0/
#  ○ To ensure response quality, we excluded respondents that were flagged by our survey
    system as “Spam” or "Duplicate. We also dropped responses from respondents that
    spent less than 2 minutes completing the survey, and dropped responses from
    respondents that selected fewer than 15 answer choices in total.
#  ○ To protect the respondents’ privacy, free-form text responses were not included in the
    public survey dataset, and the order of the rows was shuffled (responses are not
    displayed in chronological order). If a country or territory received less than 50
    respondents, we grouped them into a group named “Other”, again for the purpose of
    de-identification.

Q1
Welcome to the 2022 Kaggle Machine Learning and Data Science Survey! It should take roughly 10 to 15
minutes to complete this survey. Anonymized survey results will be released publicly at the end of the
year.
Q2
What is your age (# years)?
[List of Values]
Q3
What is your gender?
● Man
● Woman
● Nonbinary
● Prefer not to say
● Prefer to self-describe
Q4
In which country do you currently reside?
[List of Countries]
Q5
Are you currently a student? (high school, university, or graduate)
● Yes
● No
Q6
On which platforms have you begun or completed data science courses? (Select all that apply)
● Coursera
● edX
● Kaggle Learn Courses
● DataCamp
● Fast.ai
● Udacity
● Udemy
● LinkedIn Learning
● Cloud-certification programs (direct from AWS, Azure, GCP, or similar)
● University Courses (resulting in a university degree)
● None
● Other
Q7
What products or platforms did you find to be most helpful when you first started studying data
science? (Select all that apply)
● University courses
● Online courses (Coursera, EdX, etc)
● Social media platforms (Reddit, Twitter, etc)
● Video platforms (YouTube, Twitch, etc)
● Kaggle (notebooks, competitions, etc)
● None / I do not study data science
● Other
Q8
What is the highest level of formal education that you have attained or plan to attain within the next 2
years?
● No formal education past high school
● Some college/university study without earning a bachelor’s degree
● Bachelor’s degree
● Master’s degree
● Doctoral degree
● Professional doctorate
● I prefer not to answer
Q92
Have you ever published any academic research (papers, preprints, conference proceedings, etc)?
● Yes
● No
Q103
Did your research make use of machine learning? (select multiple)
● Yes, the research made advances related to some novel machine learning method (theoretical
research)
● Yes, the research made use of machine learning as a tool (applied research)
● No
Q11
For how many years have you been writing code and/or programming?
● I have never written code
● < 1 years
● 1-2 years
● 3-5 years
● 5-10 years
● 10-20 years
● 20+ years
Q124
What programming languages do you use on a regular basis? (Select all that apply)
● Python
● R
● SQL
● C
● C++
● Java
● Javascript
● Julia
● Bash
● MATLAB
● None
● Other
● C#
● PHP
● Go
Q135
Which of the following integrated development environments (IDE's) do you use on a regular basis?
(Select all that apply)
● JupyterLab
● RStudio / Posit
● Visual Studio
● Visual Studio Code (VSCode)
● PyCharm
● Spyder
● Notepad++
● Sublime Text
● Vim, Emacs, or similar
● MATLAB
● Jupyter Notebook
● IntelliJ
● None
● Other
Q146
Do you use any of the following hosted notebook products? (Select all that apply)
● Kaggle Notebooks
● Colab Notebooks
● Azure Notebooks
● Code Ocean
● IBM Watson Studio
● Amazon Sagemaker Studio
● Amazon Sagemaker Studio Lab
● Amazon EMR Notebooks
● Google Cloud Vertex AI Workbench
● Hex Workspaces
● Noteable Notebooks
● Databricks Collaborative Notebooks
● Deepnote Notebooks
● Gradient Notebooks
● None
● Other
Q157
Do you use any of the following data visualization libraries on a regular basis? (Select all that apply)
● Matplotlib
● Seaborn
● Plotly / Plotly Express
● Ggplot / ggplot2
● Shiny
● D3 js
● Altair
● Bokeh
● Geoplotlib
● Leaflet / Folium
● Pygal
● Dygraphs
● Highcharter
● None
● Other
Q168
For how many years have you used machine learning methods?
● I do not use machine learning methods
● Under 1 year
● 1-2 years
● 2-3 years
● 3-4 years
● 4-5 years
● 5-10 years
● 10-20 years
● 20 or more years
Q179
Which of the following machine learning frameworks do you use on a regular basis? (Select all that
apply)
● Scikit-learn
● TensorFlow
● Keras
● PyTorch
● Fast.ai
● Xgboost
● LightGBM
● CatBoost
● Caret
● Tidymodels
● JAX
● PyTorch Lightning
● Huggingface
● None
● Other
Q1810
Which of the following ML algorithms do you use on a regular basis? (Select all that apply):
● Linear or Logistic Regression
● Decision Trees or Random Forests
● Gradient Boosting Machines (xgboost, lightgbm, etc)
● Bayesian Approaches
● Evolutionary Approaches
● Dense Neural Networks (MLPs, etc)
● Convolutional Neural Networks
● Generative Adversarial Networks
● Recurrent Neural Networks
● Transformer Networks (BERT, gpt-3, etc)
● Autoencoder Networks (DAE, VAE, etc)
● Graph Neural Networks
● None
● Other
Q1911
Which categories of computer vision methods do you use on a regular basis? (Select all that apply)
● General purpose image/video tools (PIL, cv2, skimage, etc)
● Image segmentation methods (U-Net, Mask R-CNN, etc)
● Object detection methods (YOLOv6, RetinaNet, etc)
● Image classification and other general purpose networks (VGG, Inception, ResNet,
ResNeXt, NASNet, EfficientNet, etc)
● Vision transformer networks (ViT, DeiT, BiT, BEiT, Swin, etc)
● Generative Networks (GAN, VAE, etc)
● None
● Other
Q20
Which of the following natural language processing (NLP) methods do you use on a regular basis?
(Select all that apply)
● Word embeddings/vectors (GLoVe, fastText, word2vec)
● Encoder-decoder models (seq2seq, vanilla transformers)
● Contextualized embeddings (ELMo, CoVe)
● Transformer language models (GPT-3, BERT, XLnet, etc)
● None
● Other
Q21
Do you download pre-trained model weights from any of the following services? (Select all that apply)
● Tfhub.dev
● Pytorch hub
● Huggingface models
● Timm
● Jumpstart
● ONNX models
● NVIDIA NGC models
● Kaggle datasets
● Other storage services (i.e. google drive)
● I do not download pre-trained model weights on a regular basis
Q22
Which of the following ML model hubs/repositories do you use most often? (Select all that apply)
● » Tfhub.dev
● » Pytorch hub
● » Huggingface models
● » Timm
● » Jumpstart
● » ONNX models
● » NVIDIA NGC models
● » Kaggle datasets
● » Other storage services (i.e. google drive)
Q23
Select the title most similar to your current role (or most recent title if retired):
● Data Analyst (Business, Marketing, Financial, Quantitative, etc)
● Data Architect
● Data Engineer
● Data Scientist
● Data Administrator
● Developer Advocate
● Machine Learning/ MLops Engineer
● Manager (Program, Project, Operations, Executive-level, etc)
● Research Scientist
● Software Engineer
● Engineer (non-software)
● Statistician
● Teacher / professor
● Currently not employed
● Other
15 If
Q24
In what industry is your current employer/contract (or your most recent employer if retired)?
● Academics/Education
● Accounting/Finance
● Broadcasting/Communications
● Computers/Technology
● Energy/Mining
● Government/Public Service
● Insurance/Risk Assessment
● Online Service/Internet-based Services
● Marketing/CRM
● Manufacturing/Fabrication
● Medical/Pharmaceutical
● Non-profit/Service
● Retail/Sales
● Shipping/Transportation
● Other
Q25
What is the size of the company where you are employed?
● 0-49 employees
● 50-249 employees
● 250-999 employees
● 1000-9,999 employees
● 10,000 or more employees
Q26
Approximately how many individuals are responsible for data science workloads at your place of
business?
● 0
● 1-2
● 3-4
● 5-9
● 10-14
● 15-19
● 20+
Q27
Does your current employer incorporate machine learning methods into their business?
● We are exploring ML methods (and may one day put a model into production)
● We use ML methods for generating insights (but do not put working models into production)
● We recently started using ML methods (i.e., models in production for less than 2 years)
● We have well established ML methods (i.e., models in production for more than 2 years)
● No (we do not use ML methods)
● I do not know
Q28
Select any activities that make up an important part of your role at work: (Select all that apply)
● Analyze and understand data to influence product or business decisions
● Build and/or run the data infrastructure that my business uses for storing, analyzing, and
operationalizing data
● Build prototypes to explore applying machine learning to new areas
● Build and/or run a machine learning service that operationally improves my product or
workflows
● Experimentation and iteration to improve existing ML models
● Do research that advances the state of the art of machine learning
● None of these activities are an important part of my role at work
● Other
Q29
What is your current yearly compensation (approximate $USD)?
[List of Values]
Q30
Approximately how much money have you spent on machine learning and/or cloud computing
services at home or at work in the past 5 years (approximate $USD)?
● $0 ($USD)
● $1-$99
● $100-$999
● $1000-$9,999
● $10,000-$99,999
● $100,000 or more ($USD)
Q31
Which of the following cloud computing platforms do you use? (Select all that apply)
● Amazon Web Services (AWS)
● Microsoft Azure
● Google Cloud Platform (GCP)
● IBM Cloud / Red Hat
● Oracle Cloud
● SAP Cloud
● VMware Cloud
● Alibaba Cloud
● Tencent Cloud
● Huawei Cloud
● None
● Other
Q32
Of the cloud platforms that you are familiar with, which has the best developer experience (most
enjoyable to use)?
● » Amazon Web Services (AWS)
● » Microsoft Azure
● » Google Cloud Platform (GCP)
● » IBM Cloud / Red Hat
● » Oracle Cloud
● » SAP Cloud
● » Salesforce Cloud
● » VMware Cloud
● » Alibaba Cloud
● » Tencent Cloud
● » Huawei Cloud
● None were satisfactory
● They all had a similarly enjoyable developer experience
● Other
Q33
Do you use any of the following cloud computing products? (Select all that apply)
● Amazon Elastic Compute Cloud (EC2)
● Microsoft Azure Virtual Machines
● Google Cloud Compute Engine
● No / None
● Other
Q34
Do you use any of the following data storage products? (Select all that apply)
● Amazon Simple Storage Service (S3)
● Amazon Elastic File System (EFS)
● Google Cloud Storage (GCS)
● Google Cloud Filestore
● Microsoft Azure Blob Storage
● Microsoft Azure Files
● No / None
● Other
Q35
Do you use any of the following data products (relational databases, data warehouses, data lakes,
or similar)? (Select all that apply)
● MySQL
● PostgreSQL
● SQLite
● Oracle Database
● MongoDB
● Snowflake
● IBM Db2
● Microsoft SQL Server
● Microsoft Azure SQL Database
● Amazon Redshift
● Amazon RDS
● Amazon DynamoDB
● Google Cloud BigQuery
● Google Cloud SQL
● None
● Other
Q36
Do you use any of the following business intelligence tools? (Select all that apply)
● Amazon QuickSight
● Microsoft Power BI
● Google Data Studio
● Looker
● Tableau
● Qlik Sense
● Domo
● TIBCO Spotfire
● Alteryx
● Sisense
● SAP Analytics Cloud
● Microsoft Azure Synapse
● Microstrategy
● None
● Other
Q37
Do you use any of the following managed machine learning products? (Select all that apply)
● Amazon SageMaker
● Azure Machine Learning Studio
● Google Cloud Vertex AI
● DataRobot
● Databricks
● Dataiku
● Alteryx
● Rapidminer
● C3.ai
● Domino Data Lab
● H2O AI Cloud
● No / None
● Other
Q38
Do you use any of the following automated machine learning tools? (Select all that apply)
● Google Cloud AutoML
● H20 Driverless AI
● Databricks AutoML
● DataRobot AutoML
● Amazon Sagemaker Autopilot
● Azure Automated Machine Learning
● No / None
● Other
Q39
Do you use any of the following products to serve your machine learning models? (Select all that
apply)
● TensorFlow Extended (TFX)
● TorchServe
● ONNX Runtime
● Triton Inference Server
● OpenVINO Model Server
● KServe
● BentoML
● Multi Model Server (MMS)
● Seldon Core
● MLflow
● Other
● None
Q40
Do you use any tools to help monitor your machine learning models and/or experiments? (Select all
that apply)
● Neptune.ai
● Weights & Biases
● Comet.ml
● TensorBoard
● Guild.ai
● ClearML
● MLflow
● Aporia
● Evidently AI
● Arize
● WhyLabs
● Fiddler
● DVC
● No / None
● Other
32 If
Q41
Do you use any of the following responsible or ethical AI products in your machine learning
practices? (Select all that apply)
● Google Responsible AI Toolkit (LIT, What-If Tools, Fairness Indicator, TensorFlow Data
Validation, TensorFlow Privacy, etc.)
● Microsoft Responsible AI Toolbox (Fairlearn, Counterfit, InterpretML, SmartNoise, etc.)
● IBM AI Ethics tools (AI Fairness 360, Adversarial Robustness Toolbox, AI Explainability 360,
etc.)
● Amazon AI Ethics Tools (Clarify, A2I, etc)
● The LinkedIn Fairness Toolkit (LiFT)
● Audit-AI
● Aequitas
● None
● Other
Q42
Do you use any of the following types of specialized hardware when training machine learning
models? (Select all that apply)
● GPUs
● TPUs
● IPUs
● WSEs
● RDUs
● Trainium Chips
● Inferentia Chips
● None
● Other
Q43
Approximately how many times have you used a TPU (tensor processing unit)?
● Never
● Once
● 2-5 times
● 6-25 times
● More than 25 times
Q44
Who/what are your favorite media sources that report on data science topics? (Select all that apply)
● Twitter (data science influencers)
● Email newsletters (Data Elixir, O'Reilly Data & AI, etc)
● Reddit (r/machinelearning, etc)
● Kaggle (notebooks, forums, etc)
● Course Forums (forums.fast.ai, Coursera forums, etc)
● YouTube (Kaggle YouTube, Cloud AI Adventures, etc)
● Podcasts (Chai Time Data Science, O’Reilly Data Show, etc)
● Blogs (Towards Data Science, Analytics Vidhya, etc)
● Journal Publications (peer-reviewed journals, conference proceedings, etc)
● Slack Communities (ods.ai, kagglenoobs, etc)
● None
● Other
