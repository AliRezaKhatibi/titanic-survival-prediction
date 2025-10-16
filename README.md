## 📖 **Project Summary**

**🚢 Dataset:** [Kaggle Titanic](https://www.kaggle.com/c/titanic)  
- **891 passengers** (train) + **418 test**  
- **12 features**: Pclass, Sex, Age, Fare, Cabin, etc.  
- **Target**: Survived (0/1) - **38.4% survival rate**

**🔧 Method:** Complete End-to-End ML Pipeline  
| Step | Description | Output |
|------|-------------|--------|
| **1. EDA** | 15+ visualizations (survival by class/sex/age) | Insights |
| **2. Cleaning** | Handle 177 Age + 687 Cabin missing | Clean data |
| **3. Features** | FamilySize, AgeBin(5), FareBin(4), Has_Cabin, IsAlone | +8.2% boost |
| **4. Models** | Logistic + RF + SVM + **XGBoost** | 4 trained |
| **5. Tuning** | GridSearchCV + RandomizedSearchCV (30 iter) | Optimal params |
| **6. Eval** | 5-Fold CV + F1/AUC/ROC/Confusion Matrix | Metrics |

**🏆 Results Comparison:**

| Model | CV Accuracy | Val Accuracy | **F1 Score** | **AUC** | Time | 🥇 |
|-------|-------------|--------------|--------------|---------|------|----|
| **Random Forest** | **82.73%** | **82.68%** | **0.7832** | **0.9033** | 64s | 🏆 |
| XGBoost | **83.71%** | 79.89% | 0.7391 | 0.8943 | 13s | ⭐ |
| SVM | 82.72% | 81.01% | 0.7606 | 0.8638 | 8s | - |
| Logistic | 80.33% | 81.56% | 0.7692 | 0.8970 | 3s | - |

**🔥 Key Insights:**
- **Sex**: 32% importance (Women: 74% survival)
- **Pclass**: 8% (1st Class: 63%)
- **FamilySize**: +4.8% boost
- **Test Predictions**: **140/418 survived** (33.5%)

**✅ Output:** `titanic_submission.csv` - **Kaggle Score: 0.78-0.82** (Top 20%)  
**⏱️ Total Runtime:** 5 minutes | **📁 Files:** 15+ plots + models