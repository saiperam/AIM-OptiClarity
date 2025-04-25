# OptiClarity 👁️💡  
**AI-Powered Eye Disease Diagnosis and Vision Deficiency Assessment**  

## 🚀 Overview  
OptiClarity is an AI-powered eye health tool that combines deep learning, real-time guidance, and intelligent scheduling. It leverages four deep learning models, including an ensemble model, for precise eye scan analysis. A Langchain-based chatbot, using a RAG pipeline, provides expert guidance, while a backtracking algorithm ensures conflict-free appointment scheduling for ophthalmologists and optometrists.  

## 🏆 Features  
✅ Deep Learning for Eye Scan Analysis – OptiClarity leverages four deep learning models, including an ensemble model that combines the strengths of individual models to enhance diagnostic accuracy. The tool is capable of analyzing various eye scans, such as OCT, Fundus, Slit Lamp, and Corneal Topography, for comprehensive eye health assessments.
✅ Smart AI Chatbot – Langchain-based chatbot using a RAG pipeline for real-time, expert-level guidance.
✅ Conflict-Free Appointment Scheduling – Intelligent scheduling using a backtracking algorithm to ensure seamless booking for ophthalmologists and optometrists.
✅ Sleek and Mobile-Responsive UI – A user-friendly interface that is optimized for both web and mobile, ensuring smooth and efficient patient interaction. 

## 🔬 Technology Stack  
- **Deep Learning**: TensorFlow, Keras, PyTorch, CNN architectures
- **Models**: ResNet50, MobileNetV2, InceptionV3, EfficientNetB3, Miistral7B
- **Medical Imaging Preprocessing**: Scikit-learn, NumPy, Pandas, Matplotlib, OpenCV
- **Backend**: Flask, FastAPI, Python 
- **Frontend**: React.js, Lucide, Tailwind CSS 
- **Cloud & Deployment**: AWS S3, Docker  

## 🔬 How It Works  
1️⃣ **Input 1**: Upload Scan images in dropbox Or Enter query in Optibot.
2️⃣ **Step 1**: **Deep Learning Model** extract patterns from eye scans by learning hierarchical features through convolutional layers to identify and classify eye conditions with high accuracy.  
3️⃣ **Step 2**: **RAG LangChain Optibot**  retrieves relevant information from an expert knowledge base and generates context-aware responses by combining retrieval with a language model, enabling real-time, reliable guidance.

## 🎯 Impact & Industry Validation  
OptiClarity accelerates and enhances eye care, enabling:
✅ Instant bridging of the gap between early symptoms and diagnosis, potentially saving vision.
✅ AI-driven, multi-scan analysis that supports ophthalmologists in making faster, more accurate clinical decisions.
✅ Seamless integration of AI into real-world workflows, improving both patient experience and clinical efficiency.

## 🛠️ Getting Started  

### 🔹 Prerequisites  
- Python **3.8+**  
- Node.js **v23.4.0**  
- TensorFlow & PyTorch installed

### 🔹 Installation  

#### **1️⃣ Clone the Repository**  
```bash
git clone https://github.com/YourUsername/AIM-OptiClarity.git
cd AIM-OptiClarity
```
#### **2️⃣ Set Up the Frontend**
```bash
npm install
npm run dev
```
#### **3️⃣ Set Up the Backend**
```bash
cd backend
pip install -r requirements.txt
run all the .py files
```

