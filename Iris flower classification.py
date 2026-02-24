{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 109,
   "id": "c806c161-0a51-4052-82ef-6a2232cbfdc0",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.svm import SVC\n",
    "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 110,
   "id": "d684f739-629a-44fa-a8d9-0246f1a56aba",
   "metadata": {},
   "outputs": [],
   "source": [
    "df=pd.read_csv(r\"D:\\Arnita\\Iris.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 111,
   "id": "a8b921c6-94b3-424d-8f43-78d3459a1108",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Id</th>\n",
       "      <th>SepalLengthCm</th>\n",
       "      <th>SepalWidthCm</th>\n",
       "      <th>PetalLengthCm</th>\n",
       "      <th>PetalWidthCm</th>\n",
       "      <th>Species</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>5.1</td>\n",
       "      <td>3.5</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "      <td>Iris-setosa</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>4.9</td>\n",
       "      <td>3.0</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "      <td>Iris-setosa</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>4.7</td>\n",
       "      <td>3.2</td>\n",
       "      <td>1.3</td>\n",
       "      <td>0.2</td>\n",
       "      <td>Iris-setosa</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>4.6</td>\n",
       "      <td>3.1</td>\n",
       "      <td>1.5</td>\n",
       "      <td>0.2</td>\n",
       "      <td>Iris-setosa</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>5.0</td>\n",
       "      <td>3.6</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "      <td>Iris-setosa</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Id  SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm      Species\n",
       "0   1            5.1           3.5            1.4           0.2  Iris-setosa\n",
       "1   2            4.9           3.0            1.4           0.2  Iris-setosa\n",
       "2   3            4.7           3.2            1.3           0.2  Iris-setosa\n",
       "3   4            4.6           3.1            1.5           0.2  Iris-setosa\n",
       "4   5            5.0           3.6            1.4           0.2  Iris-setosa"
      ]
     },
     "execution_count": 111,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 112,
   "id": "ef8fc93c-c8a9-4bd2-a34b-10bb9916d6f0",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.DataFrame'>\n",
      "RangeIndex: 150 entries, 0 to 149\n",
      "Data columns (total 6 columns):\n",
      " #   Column         Non-Null Count  Dtype  \n",
      "---  ------         --------------  -----  \n",
      " 0   Id             150 non-null    int64  \n",
      " 1   SepalLengthCm  150 non-null    float64\n",
      " 2   SepalWidthCm   150 non-null    float64\n",
      " 3   PetalLengthCm  150 non-null    float64\n",
      " 4   PetalWidthCm   150 non-null    float64\n",
      " 5   Species        150 non-null    str    \n",
      "dtypes: float64(4), int64(1), str(1)\n",
      "memory usage: 7.2 KB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 113,
   "id": "0a5ba50d-2de0-4442-9ba4-c9dc7f5384e1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Species\n",
       "Iris-setosa        50\n",
       "Iris-versicolor    50\n",
       "Iris-virginica     50\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 113,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[\"Species\"].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 114,
   "id": "41e1de05-4e6f-4043-9033-d31945613ae5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Id               0\n",
       "SepalLengthCm    0\n",
       "SepalWidthCm     0\n",
       "PetalLengthCm    0\n",
       "PetalWidthCm     0\n",
       "Species          0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 114,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isna().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 115,
   "id": "a2fa47fa-fd3b-46c3-9b23-f9b19f5d6a78",
   "metadata": {},
   "outputs": [],
   "source": [
    "x=df.drop(columns=[\"Species\"])\n",
    "y=df[\"Species\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 124,
   "id": "978f1e16-0c97-4b9f-90ee-8ad79f456b49",
   "metadata": {},
   "outputs": [],
   "source": [
    "x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 125,
   "id": "1f9bdffb-dba6-4c1d-8a99-1dc7cdac9472",
   "metadata": {},
   "outputs": [],
   "source": [
    "sc=StandardScaler()\n",
    "x_train_scaled=sc.fit_transform(x_train)\n",
    "x_test_scaled=sc.fit_transform(x_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 126,
   "id": "08f063c0-a37e-4ea5-b306-1a04db2856c9",
   "metadata": {},
   "outputs": [],
   "source": [
    "model=LogisticRegression()\n",
    "model.fit(x_train_scaled, y_train)\n",
    "lr_pred=model.predict(x_test_scaled)\n",
    "lr_accuracy=accuracy_score(y_test, lr_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 127,
   "id": "692f3dea-a8c8-4b5f-876c-02e6f29232e0",
   "metadata": {},
   "outputs": [],
   "source": [
    "model=RandomForestClassifier()\n",
    "model.fit(x_train, y_train)\n",
    "rf_pred=model.predict(x_test)\n",
    "rf_accuracy=accuracy_score(y_test, rf_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 128,
   "id": "6f2da340-3450-4844-b90b-e91dd8370a0b",
   "metadata": {},
   "outputs": [],
   "source": [
    "model=SVC()\n",
    "model.fit(x_train_scaled, y_train)\n",
    "svm_pred=model.predict(x_test_scaled)\n",
    "svm_accuracy=accuracy_score(y_test, svm_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 129,
   "id": "ee78d393-d8d2-496a-acbb-993e92041cc0",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Logistic Regression Accuracy: 0.9333333333333333\n",
      "Random Forest Accuracy: 1.0\n",
      "SVM Accuracy: 0.9\n"
     ]
    }
   ],
   "source": [
    "print(\"Logistic Regression Accuracy:\", lr_accuracy)\n",
    "print(\"Random Forest Accuracy:\", rf_accuracy)\n",
    "print(\"SVM Accuracy:\", svm_accuracy)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 130,
   "id": "f45e7498-f38e-403a-83ec-72657a4a797f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Logistic Regression Report:\n",
      "                  precision    recall  f1-score   support\n",
      "\n",
      "    Iris-setosa       1.00      1.00      1.00        11\n",
      "Iris-versicolor       1.00      0.85      0.92        13\n",
      " Iris-virginica       0.75      1.00      0.86         6\n",
      "\n",
      "       accuracy                           0.93        30\n",
      "      macro avg       0.92      0.95      0.92        30\n",
      "   weighted avg       0.95      0.93      0.94        30\n",
      "\n",
      "Random Forest Report:\n",
      "                  precision    recall  f1-score   support\n",
      "\n",
      "    Iris-setosa       1.00      1.00      1.00        11\n",
      "Iris-versicolor       1.00      1.00      1.00        13\n",
      " Iris-virginica       1.00      1.00      1.00         6\n",
      "\n",
      "       accuracy                           1.00        30\n",
      "      macro avg       1.00      1.00      1.00        30\n",
      "   weighted avg       1.00      1.00      1.00        30\n",
      "\n",
      "SVM Report:\n",
      "                  precision    recall  f1-score   support\n",
      "\n",
      "    Iris-setosa       1.00      1.00      1.00        11\n",
      "Iris-versicolor       1.00      0.77      0.87        13\n",
      " Iris-virginica       0.67      1.00      0.80         6\n",
      "\n",
      "       accuracy                           0.90        30\n",
      "      macro avg       0.89      0.92      0.89        30\n",
      "   weighted avg       0.93      0.90      0.90        30\n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(\"Logistic Regression Report:\\n\", classification_report(y_test, lr_pred))\n",
    "print(\"Random Forest Report:\\n\", classification_report(y_test, rf_pred))\n",
    "print(\"SVM Report:\\n\", classification_report(y_test, svm_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 131,
   "id": "7b76cf05-99dc-4d11-8b42-4edac28b897c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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BT399NPmgqDDp9apU0c2b97sSznqhC/6s5ZBF12nDh48KE2bNpVMmTKZ32vVqpX8/fffvu3q79x7772SOXNm87wOvbp+/Xrz3IkTJ+Sxxx6TggULmrHdK1asKLNnz5aUItif8+7du83vRp+hb9SoUVK8eHHfz1u3bpUGDRqYz0wHmHn88cfNjH/xlUun19GsVOHChc04IwUKFJAXXnghzqoIPZ86depktq/lrlChgixatMj3/Lx586R8+fJmW/q777zzTrzH6mbnmZZNB8358MMPpVixYmaf4c6O8yN61Ya3umHhwoVSrlw583npZ6HZhkaNGpkLvx7vWbNmxTgH/Mv5+++/m5/nz59vvg/0771y5cpRpqWOrWrjyy+/lOrVq5v3psegefPmvudmzJhhRl7U7xYNWlq3bi3Hjh1L8nFH8iCQSAbTpk0zdwc///yzTJo0KcbzY8eONX/cn3zyiezatUtmzpxp/pDjcrPXP/zww+aPcPHixWYu+qpVq0rdunXl5MmTJuXYq1cv88WvXyC66Dodf12/3PU1P/zwgyxdulT2799vnvNq06aN3HLLLSb1rdvt27evpEmTxjx36dIlE1h89dVX5uL2zDPPmIva2rVrJaUI5udcqlQp88Wqr/GnP+uXrNILhAaJVapUMQGdjlanF2S9MMdXLr3w6wXnvffekz179pgLhAZ+sdHzQgMV/d2PP/5Ytm/fLsOGDZNUqVKZ5/U80P09+uij8ttvv5kgYMCAAb7gNLbt3ew8U3v37jXl1IuVW6rgkvv8iM2FCxdk+PDhJkjbtm2b5MmTR5544gn566+/TOChx/z9999P0EX85Zdflt69e5vPR8ujNxLXrl2L9bX6vaCBg1Z3bNy40cz1oMM1e129elVee+01c7Oi56MGK1o1hDChs38iuNq1a+dp2rSpeVy7dm1PlSpVYrxGD/3nn39uHj///POeOnXqeG7cuJGg7cf3+h9//NGTJUsWz6VLl6KsL168uOe9994zjwcNGuSpXLlylOe//fZbT6pUqTwHDx70rdu2bZsp59q1a83PmTNn9nz00UeehGrUqJGnV69eHrcK9ec8atQo87l57dq1y2xvx44d5ufXXnvNU79+/Si/c+jQIfMafW1c5XrnnXc8pUqV8ly5ciXW/RYpUsTsW33zzTeeiIgI3/aia926tee+++6Lsq5P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",
      "text/plain": [
       "<Figure size 640x480 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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t2rDW92wcOFCKV++vPm+9LvQbEOzZs3MiV/395w5c647BvzL+eeff5r7n376qfk90L/3KlWqxJqWOr6qjc8//1xuu+0289l0H9xzzz2+x2bNmmVGXtTfFg1a2rdvL4cPH07xfkfqIJBIBTNmzDBXBz/++KNMmTLluscnTJhg/rg/+ugj2b59u8yePdv8ISfkRs+/7777zB/h4sWLzVz01apVk4YNG8rx48dNyrF3797mh19/QHTRdTr+uv6463O+++47Wbp0qezatcs85vXggw/KzTffbFLf+r79+vWT9OnTm8cuXLhgAosvvvjCnNyeeOIJc1Jbs2aNpBXB/J5Lly5tflj1Of70vv7IKj1BaJBYtWpVE9DpaHV6QtYTc2Ll0hO/nnDeeust2bFjhzlBaOAXHz0uNFDR177//vvy+++/y4gRIyRdunTmcT0OdHvt2rWTTZs2mSBg0KBBvuA0vve70XGmdu7cacqpJyu3VMGl9vERn/Pnz8vIkSNNkLZlyxbJmzevPPzww3LgwAETeOg+f/vtt5N0En/hhRfkueeeM9+PlkcvJK5cuRLvc/V3QQMHre5Yv369metBh2v2unz5sgwbNsxcrOjxqMGKVg0hTOjsnwiujh07elq2bGlu16tXz1O1atXrnqO7/rPPPjO3n3nmGU+DBg08165dS9L7J/b877//3pMtWzbPhQsXYq0vUaKE56233jK3hwwZ4qlSpUqsx7/++mtPunTpPHv37vWt27JliynnmjVrzP2sWbN6pk+f7kmqZs2aeXr37u1xq1B/z2PHjjXfm9f27dvN+23dutXcHzZsmKdx48axXrNv3z7zHH1uQuUaPXq0p3Tp0p5Lly7Fu92iRYuabauvvvrKExER4Xu/uNq3b++58847Y63r06ePp3z58vG+X1KOMz0+06dP7zl8+LAnnNl9fKxYscLcP3HihLk/bdo0c3/Dhg2+1+hzdd3atWt963bs2GHWeb+zuOXcvXu3uf/uu+9e9x16t63byp49u+/xWrVqeR588EFPUml59P3OnDmT5NfAPmQkUoFeqSdGI2+N6suUKWNSzF9//bXvsSeffNKkgL3LjZ6vEf3Zs2clV65csV63e/du+eOPPxIsw9atW6Vw4cJm8dL0p6Yn9THv5C9aZdKoUSNzVer/flrfqlcUemWbM2dOs82vvvrKpE7TimB/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",
      "text/plain": [
       "<Figure size 640x480 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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umHu3LlSvnx583npZ6HZhiZNmpgTvx7vqVOnxvoORC3nn3/+ae7PmjXL/B7ov/cqVapEm5Y6rqqN7777Tu666y7z3vQYtGzZ0vfYlClTzMiL+tuiQUvbtm3l+PHjyT7uSBkEEilg0qRJ5urgt99+k/Hjx8d6fNSoUeYf91dffSU7d+6UL7/80vxDjs+tnv/www+bf4QLFiwwc9FXq1ZN6tevL6dPnzYpx+7du5sffv0B0UXX6fjr+uOuz/n5559l0aJFsm/fPvOYV7t27eS2224zqW/d7uuvvy7p0qUzj125csUEFvPnzzcnt+eee86c1FatWiWpRSA/5zJlypgfVn1OVHpff2SVniA0SKxataoJ6HS0Oj0h64k5oXLpiV9POB999JHs3r3bnCA08IuLfi80UNHXfvHFF7Jt2zYZNGiQpEmTxjyu3wPd36OPPiqbN282QUCfPn18wWlc27vV90zt2bPHlFNPVm6pgkvp70dcLl++LIMHDzZB2tatWyVfvnzy5JNPypEjR0zgocf8448/TtRJ/I033pAePXqYz0fLoxcSN27ciPO5+ruggYNWd6xfv97M9aDDNXtdv35d3n77bXOxot9HDVa0agghQmf/RGBFRER4mjdvbm7XrVvXU7Vq1VjP0UP/7bffmtsvvfSSp169ep7IyMhEbT+h5//yyy+ebNmyea5cuRJtfcmSJT0fffSRud2vXz9PlSpVoj3+448/etKkSeM5ePCgb93WrVtNOVetWmXuZ82a1fP55597EqtJkyae7t27e9wq2J/z8OHDzefmtXPnTrO97du3m/tvv/22p2HDhtFec+jQIfMcfW585Ro6dKinTJkynmvXrsW532LFipl9qx9++METFhbm215Mbdu29dx///3R1vXs2dNTvnz5OLeXmO+Zfj/TpUvnOX78uCeU2f39WLZsmbl/5swZc3/ixInm/oYNG3yv0efqutWrV/vW7d6926zzfmYxy7l//35z/9NPP431GXr3rfvKnj277/FatWp52rVr50ksLY9u78KFC4l+DexDRiIF6JV6QjTy1qi+bNmyJsX8448/+h57/vnnTQrYu9zq+RrRX7x4UXLnzh3tdfv375e9e/fGW4bt27dLkSJFzOKl6U9NT+pj3slftMqkQYMG5qo06va0vlWvKPTKNleuXGafP/zwg0mdphaB/pz1Kl+v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",
      "text/plain": [
       "<Figure size 640x480 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Logistic Regression \n",
    "cm_lr = confusion_matrix(y_test, lr_pred)\n",
    "\n",
    "plt.figure()\n",
    "sns.heatmap(cm_lr, annot=True, fmt=\"d\", cmap=\"Blues\",\n",
    "            xticklabels=model.classes_,\n",
    "            yticklabels=model.classes_)\n",
    "plt.xlabel(\"Predicted\")\n",
    "plt.ylabel(\"Actual\")\n",
    "plt.title(\"Confusion Matrix - Logistic Regression\")\n",
    "plt.show()\n",
    "\n",
    "\n",
    "# Random Forest \n",
    "cm_rf = confusion_matrix(y_test, rf_pred)\n",
    "\n",
    "plt.figure()\n",
    "sns.heatmap(cm_rf, annot=True, fmt=\"d\", cmap=\"Blues\",\n",
    "            xticklabels=model.classes_,\n",
    "            yticklabels=model.classes_)\n",
    "plt.xlabel(\"Predicted\")\n",
    "plt.ylabel(\"Actual\")\n",
    "plt.title(\"Confusion Matrix - Random Forest\")\n",
    "plt.show()\n",
    "\n",
    "\n",
    "# SVM \n",
    "cm_svm = confusion_matrix(y_test, svm_pred)\n",
    "\n",
    "plt.figure()\n",
    "sns.heatmap(cm_svm, annot=True, fmt=\"d\", cmap=\"Blues\",\n",
    "            xticklabels=model.classes_,\n",
    "            yticklabels=model.classes_)\n",
    "plt.xlabel(\"Predicted\")\n",
    "plt.ylabel(\"Actual\")\n",
    "plt.title(\"Confusion Matrix - SVM\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a9a7556a-7294-486e-b61d-560f38d85755",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
