# Hyperparameter Tuning Methods for Batik Classification

This directory contains three different hyperparameter tuning approaches for the batik classification model:

## 📁 Directory Structure

```
hyperparameterTuning/
├── randomSearch/
│   └── RandomSearch.ipynb          # Random Search implementation
├── gridSearch/
│   └── GridSearch.ipynb            # Grid Search implementation
├── geneticAlgorithm/
│   └── GeneticAlgorithm.ipynb      # Genetic Algorithm implementation
└── README.md                       # This file
```

## 🔍 Hyperparameter Tuning Methods

### 1. Random Search (`randomSearch/RandomSearch.ipynb`)

**What it does:**
- Randomly samples hyperparameter combinations from the search space
- Uses Keras Tuner's `RandomSearch` implementation
- Stops after a specified number of trials

**Advantages:**
- ✅ Fast and efficient
- ✅ Good for high-dimensional search spaces
- ✅ Can find good solutions quickly
- ✅ Less likely to get stuck in local optima

**Disadvantages:**
- ❌ May miss optimal combinations
- ❌ No guarantee of finding the best solution
- ❌ Results can be inconsistent between runs

**Best for:** Quick exploration of hyperparameter space, when you have limited time

---

### 2. Grid Search (`gridSearch/GridSearch.ipynb`)

**What it does:**
- Systematically tries ALL possible combinations of hyperparameters
- Uses Keras Tuner's `GridSearch` implementation
- Exhaustive search through the parameter space

**Advantages:**
- ✅ Guaranteed to find the optimal solution (within the grid)
- ✅ Reproducible results
- ✅ Systematic and thorough
- ✅ Good for small search spaces

**Disadvantages:**
- ❌ Extremely slow for large search spaces
- ❌ Exponential growth with parameter count
- ❌ Computationally expensive
- ❌ May not scale well

**Best for:** Small search spaces, when you need guaranteed optimal results

---

### 3. Genetic Algorithm (`geneticAlgorithm/GeneticAlgorithm.ipynb`)

**What it does:**
- Implements a custom genetic algorithm for hyperparameter optimization
- Uses evolutionary principles: selection, crossover, mutation
- Maintains a population of hyperparameter sets

**Advantages:**
- ✅ Can find good solutions with fewer evaluations
- ✅ Maintains diversity in search space
- ✅ Can escape local optima through mutation
- ✅ More efficient than GridSearch for large spaces
- ✅ Inspired by biological evolution

**Disadvantages:**
- ❌ More complex implementation
- ❌ Requires tuning of genetic parameters (mutation rate, crossover rate)
- ❌ No guarantee of finding global optimum
- ❌ Results can vary between runs

**Best for:** Large search spaces, when you want efficient exploration

---

## 🎯 Hyperparameters Being Tuned

All three methods optimize the same hyperparameters:

1. **Learning Rate**: `[1e-4, 3e-4, 1e-3, 3e-3, 1e-2]`
2. **Optimizer**: `['adam', 'rmsprop', 'sgd']`
3. **Dense Units**: `[64, 128, 256, 384, 512]`
4. **Dropout Rate**: `[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]`
5. **Add Conv Layer**: `[True, False]`
6. **Conv Filters**: `[32, 64, 96, 128]`

## 📊 Comparison Table

| Method | Speed | Completeness | Scalability | Best Use Case |
|--------|-------|--------------|-------------|---------------|
| **Random Search** | ⚡ Fast | 🔍 Partial | 📈 Good | Quick exploration |
| **Grid Search** | 🐌 Slow | ✅ Complete | 📉 Poor | Small spaces |
| **Genetic Algorithm** | ⚡⚡ Medium | 🔍 Partial | 📈📈 Excellent | Large spaces |

## 🚀 How to Use

1. **Choose your method** based on your needs:
   - Need quick results? → Random Search
   - Small search space? → Grid Search  
   - Large search space? → Genetic Algorithm

2. **Run the notebook** for your chosen method:
   ```bash
   cd src/models/hyperparameterTuning/[method_name]
   jupyter notebook [MethodName].ipynb
   ```

3. **Monitor the results** and compare performance

## 📈 Expected Performance

- **Random Search**: Usually finds 80-90% of optimal performance in 20-30 trials
- **Grid Search**: Finds optimal solution but may take 100+ trials
- **Genetic Algorithm**: Usually finds 85-95% of optimal performance in 10-20 generations

## 🔧 Customization

Each notebook can be customized by modifying:
- Population size (Genetic Algorithm)
- Number of trials (Random/Grid Search)
- Search space parameters
- Training epochs per trial
- Callback configurations

## 📝 Notes

- All methods use the same model architecture (MobileNetV2 + custom layers)
- All methods use the same dataset and data augmentation
- Results are saved in separate directories for each method
- Training history and visualizations are generated for each method

## 🎯 Recommendations

- **Start with Random Search** for initial exploration
- **Use Grid Search** if you have a small, well-defined search space
- **Try Genetic Algorithm** for complex, high-dimensional optimization problems
- **Compare results** from multiple methods to ensure robustness 