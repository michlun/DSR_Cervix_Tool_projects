# Cervix: Cervical Cancer Prediction Tool 
by Francesco Cocciro and Michele Lunelli

Summary: The project aims to develop a web application with a simple graphical interface
for cervical cancer risk assessment and evaluation of Pap test images.

1. Cervical Cancer Risk Prediction: Upon selecting the first button, a page will be
loaded, prompting the user to fill in specific fields. This functionality allows users to
predict their probability of developing cervical cancer based on the input values. The
application will utilize a predictive model trained on relevant data to provide this
probability. Additionally, the project incorporates a text generator that explains the results obtained.
This generator will provide detailed explanations in natural language. By generating informative and
interpretable explanations, users will have a better understanding of the underlying
reasons behind the model's decisions, promoting trust and transparency in the
application.
2. Cell Type Evaluation from Pap Slide Images: Selecting a second button will load a page that
allows users to upload a microscope image, either of whole slides or single cells. The application will employ
trained deep neural networks to analyze the image and determine if it contains cervical cancer
cells or not. Single cells can also be selected from whole slide images for further evaulation.
The model's output will provide insights into the presence or absence of
tumor cells, assisting in early detection and diagnosis.
To enhance the interpretability of the results,
the project incorporates Explainable Artificial Intelligence (XAI) techniques.
Specifically, Grad-CAM++ maps are generated to identify which parts of the
input image are crucial in determining the outcome of the cell type.

Overall, this project aims to provide a user-friendly web application that empowers users to
assess the risk of cervical cancer and offers the ability to analyze Pap slide images for potential tumor
presence. By leveraging machine learning techniques and incorporating XAI and text
generation, the application can contribute to early detection, promote timely medical
intervention, and improve outcomes for individuals at risk of cervical cancer.

The models are trained and validated using the following datasets:
- Risk factors: https://archive.ics.uci.edu/dataset/383/cervical+cancer+risk+factors
- Microscope cell images: https://www.cs.uoi.gr/~marina/sipakmed.html

## Usage
Please run app.py with Flask: flask --app app.py run
