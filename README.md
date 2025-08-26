# Synthetic LOB GAN Project

## Overview
This project builds a Generative Adversarial Network (GAN) to generate synthetic Limit Order Book (LOB) data for cryptocurrency markets.

## Features
- Real-time LOB data collection (720 snapshots)
- Feature engineering and preprocessing
- GAN implementation with TensorFlow/Keras
- Statistical evaluation with KS-tests
- Animated visualization for LinkedIn

## Results
- Data collected: ✅ 720 BTC/USD order book snapshots
- GAN trained: ✅ Model generates synthetic samples
- Evaluation: KS-statistic = 0.8690 (target < 0.05)
- Visualization: Animated GIF created

## Files
- `collect_data.py` - LOB data collection
- `process_features.py` - Data preprocessing
- `train_gan.py` - GAN training
- `evaluate_gan.py` - Statistical evaluation
- `create_linkedin_content.py` - Visualization

## Next Steps
- Improve GAN architecture
- Hyperparameter tuning
- More training data

## How to Run
1. Collect data: `python collect_data.py`
2. Process features: `python process_features.py`
3. Train GAN: `python train_gan.py`
4. Evaluate: `python evaluate_gan.py`
5. Create visualization: `python create_linkedin_content.py`