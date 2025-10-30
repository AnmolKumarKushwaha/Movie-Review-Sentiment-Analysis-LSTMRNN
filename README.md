## Deep Learning Project: Movie Review Sentiment Analysis  

**Tech Stack:** Python | TensorFlow | Keras | NumPy | Streamlit | IMDB Dataset  

This project focuses on building a **Deep Learning-based Sentiment Analysis model** that classifies movie reviews as **positive** or **negative**. The goal was to explore the ability of **Recurrent Neural Networks (RNNs)**, specifically **Long Short-Term Memory (LSTM)** architectures, to understand context and sequential dependencies in natural language.  

The model was trained on the **IMDB Movie Review Dataset**, consisting of 50,000 preprocessed reviews represented through **integer encoding**. An **Embedding Layer** was employed to convert these encoded sequences into **dense vector representations**, enabling the LSTM network to capture semantic and contextual relationships between words.  

The architecture was optimized using the **Adam optimizer** and regularized through **dropout layers** to minimize overfitting. The training process was monitored using validation metrics and **EarlyStopping** callbacks for stable convergence.  

The final model demonstrated strong generalization with a **Training Accuracy of 94.20%**, **Validation Accuracy of 87.96%**, and **Test Accuracy of 87.36%**, reflecting robust performance and effective overfitting control.  

This project was later integrated with a **Streamlit web interface**, allowing users to input custom movie reviews and instantly receive a predicted sentiment (positive or negative) based on the trained model.  
It was also **successfully deployed on the Streamlit.io platform**, providing a user-friendly interface for live interaction with the model.
