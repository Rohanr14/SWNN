# Synapse-Weighted Neural Network (SWNN) for Time Series Prediction

This project showcases a novel Synapse-Weighted Neural Network (SWNN) implementation for time series prediction. The SWNN is designed to adapt dynamically to changing patterns in time series data, potentially outperforming traditional models in certain scenarios.

## What is SWNN?

The Synapse-Weighted Neural Network is a custom neural network architecture that incorporates dynamic synapse weights. Unlike traditional neural networks with static weights, SWNN adjusts its synaptic strengths based on recent activation patterns, allowing for more adaptive and context-sensitive predictions.

Key features of SWNN:
- Dynamic synaptic weight adjustment
- Adaptive learning to changing time series patterns
- Potential for improved performance on non-stationary time series

## Interactive Comparison Tool

To demonstrate the capabilities of SWNN, this project includes an interactive web application that allows users to compare SWNN against other popular time series models:

- Auto ARIMA
- Prophet
- LSTM

Users can upload their own time series data and visualize how SWNN performs compared to these established models.

## Requirements

- Python 3.7+
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- PMDArima
- Prophet
- TensorFlow

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/swnn.git
cd swnn
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your web browser and go to the URL provided by Streamlit (usually http://localhost:8501).

3. Upload your time series data in CSV format (date/time index and a single column of numeric values).

4. Select the models you want to compare with SWNN.

5. Adjust the number of time steps for SWNN and LSTM models using the slider.

6. View the results, including performance metrics and prediction visualizations.

## SWNN Implementation

The core SWNN implementation can be found in `swnn.py`. Key components include:

- `SynapseWeightedNeuron`: Implements a neuron with dynamic synaptic weights.
- `SynapseWeightedNeuralNetwork`: Constructs the full network architecture.
- Custom training loop with synapse weight updates.

For more details on the implementation, please refer to the comments in `swnn.py`.

## Results and Performance

The performance of SWNN can vary depending on the nature of the time series data. In general, SWNN may show advantages in:

- Adapting to sudden changes in time series patterns
- Capturing complex, non-linear relationships
- Maintaining performance over long-term predictions

However, like all models, it may not be the best choice for every scenario. The comparison tool allows users to evaluate its performance on their specific datasets.

## Contributing

Contributions to improve SWNN or the comparison tool are welcome. Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature-name`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add some feature'`)
5. Push to the branch (`git push origin feature/your-feature-name`)
6. Create a new Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to all contributors and users of this project.
- Special thanks to the creators and maintainers of the libraries used for comparison models.