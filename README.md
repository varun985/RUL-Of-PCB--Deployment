**Predicting Remaining Useful Life (RUL) of Printed Circuit Boards (PCBs)**

This project leverages a hybrid particle filter with Nelder-Mead optimization to accurately predict the remaining useful life (RUL) of Printed Circuit Boards (PCBs). The particle filter algorithm estimates parameters of the Steinberg formula, which models PCB fatigue life, and iteratively refines parameter estimates through prediction, correction, resampling, and estimation. The final estimates are further refined using Nelder-Mead optimization to minimize the sum of squared errors between predicted and actual values.

**Key Features:**

- **Particle Filter Algorithm**: Effectively handles non-linear dynamics and non-Gaussian noise.
- **Nelder-Mead Optimization**: Precisely refines parameter estimates.
- **Deployment**: The model is deployed using Flask and Render, enabling user interaction through a web interface at https://rul-of-pcb.onrender.com/.

**Try It Out:**

Visit https://rul-of-pcb.onrender.com/ to explore the model's capabilities and predict the RUL of PCBs. This innovative approach demonstrates the potential of machine learning and optimization techniques in real-world applications, particularly in industries where predicting the lifespan of critical components ensures reliability and safety.

**Unique Info:**

- **Innovative Application of Particle Filters**: Typically used for robot navigation, this project showcases the versatility and potential of particle filters in regression problems with limited datasets.
- **Real-World Impact**: Accurate prediction of PCB RUL can significantly improve the reliability and safety of critical components in various industries.

**Technical Details:**

- **Tech Stack**: Python, Pandas, NumPy, SciPy, Flask, and Render.
- **Model Performance**: Achieved a satisfactory R-square value of approximately 0.99, indicating a strong fit.

**Getting Started:**

1. **Access the Model**: Visit https://rul-of-pcb.onrender.com/.
2. **Input Parameters**: Provide the required input variables (Vibrational Frequency applied , Stress on PCB or capacitor).
3. **Predict RUL**: The model will predict the fatigue life of the PCB.

**Contributions and Feedback:**

This project is open to contributions and feedback. If you have any suggestions or improvements, please feel free to reach out.
