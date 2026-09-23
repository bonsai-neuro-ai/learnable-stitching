import numpy as np
import matplotlib.pyplot as plt

A = np.array([[1,1], [0,0], [0,0]]).transpose()
B = np.array([[0,0], [1,0], [0,1]]).transpose()
C = np.array([[0,0], [1,0], [0,1]]).transpose()

# Concatenate and compute covariance matrix
combined = np.vstack([A, C])
cov_matrix = np.cov(combined)
AxC = cov_matrix[:A.shape[0], A.shape[0]:]

combined = np.vstack([B, C])
cov_matrix = np.cov(combined)
BxC = cov_matrix[:B.shape[0], B.shape[0]:]

combined = np.vstack([A, A])
cov_matrix = np.cov(combined)
AxA = cov_matrix[:A.shape[0], A.shape[0]:]

combined = np.vstack([B, B])
cov_matrix = np.cov(combined)
BxB = cov_matrix[:B.shape[0], B.shape[0]:]

combined = np.vstack([C, C])
cov_matrix = np.cov(combined)
CxC = cov_matrix[:C.shape[0], C.shape[0]:]

AxC_f = np.linalg.norm(AxC)
BxC_f = np.linalg.norm(BxC)
cka_A =  (AxC_f * AxC_f) / (np.linalg.norm(AxA) *  np.linalg.norm(CxC))
cka_B =  (BxC_f * BxC_f) / (np.linalg.norm(BxB) *  np.linalg.norm(CxC))

A = [cka_A, 1]
B = [cka_B, 0.5]

barWidth = 0.25
fig = plt.subplots(figsize =(12, 8)) 

plt.bar([0.25, 1], A, color="C1", width=barWidth, label="Upstream A")
plt.bar([0, 0.75], B, color="C0", width=barWidth, label="Upstream B")

plt.xlabel('Similarity Method', fontweight ='bold', fontsize = 24) 
plt.ylabel('Similarity Measure', fontweight ='bold', fontsize = 24) 

plt.xticks( [0.125, 0.75+0.125] ,['Activity-Correlation', 'Neural Stitching'], fontsize = 20)
plt.yticks(fontsize = 20)


plt.legend(loc='upper center', fontsize=15)
plt.savefig(f"analysis/plots/toy_bar_graph.svg")