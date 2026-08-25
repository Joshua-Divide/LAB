from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import Perceptron
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data_banknote_authentication.txt'
OUT = ROOT / 'Experiment 1.pdf'
PLOTS = ROOT / '.experiment1_plots'
PLOTS.mkdir(exist_ok=True)
FONT_DIR = Path(matplotlib.get_data_path()) / 'fonts' / 'ttf'
pdfmetrics.registerFont(TTFont('DejaVuSerif', str(FONT_DIR / 'DejaVuSerif.ttf')))
pdfmetrics.registerFont(TTFont('DejaVuSerif-Bold', str(FONT_DIR / 'DejaVuSerif-Bold.ttf')))

columns = ['Variance', 'Skewness', 'Curtosis', 'Entropy', 'Class']
features = columns[:-1]
df = pd.read_csv(DATA, header=None, names=columns)

class ScratchPerceptron:
    def __init__(self, learning_rate=0.01, max_epochs=100):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.weights = np.zeros(X.shape[1], dtype=float)
        self.bias = 0.0
        self.error_history = []
        self.weight_history = []
        self.bias_history = []
        for _ in range(self.max_epochs):
            errors = 0
            for xi, yi in zip(X, y):
                pred = 1 if np.dot(self.weights, xi) + self.bias >= 0 else 0
                update = self.learning_rate * (yi - pred)
                if update != 0:
                    errors += 1
                self.weights += update * xi
                self.bias += update
            self.error_history.append(errors)
            self.weight_history.append(self.weights.copy())
            self.bias_history.append(self.bias)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return (X @ self.weights + self.bias >= 0).astype(int)

X = df[features].to_numpy(dtype=float)
y = df['Class'].to_numpy(dtype=int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = ScratchPerceptron(0.01, 100).fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

metrics = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'F1-score': f1_score(y_test, y_pred)
}

def savefig(name):
    path = PLOTS / name
    plt.savefig(path, dpi=170, bbox_inches='tight')
    plt.close()
    return path

fig, axes = plt.subplots(2, 2, figsize=(9, 6.2))
for ax, feature in zip(axes.ravel(), features):
    ax.hist(df[feature], bins=30, edgecolor='black', linewidth=0.3)
    ax.set_title(feature, fontsize=9)
    ax.set_xlabel('Value', fontsize=8)
    ax.set_ylabel('Frequency', fontsize=8)
fig.suptitle('Feature Histograms', fontsize=11)
fig.tight_layout()
hist_path = savefig('01_histograms.png')

corr = df.corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(6.2, 5.4))
im = ax.imshow(corr, vmin=-1, vmax=1, cmap='viridis')
ax.set_xticks(np.arange(len(corr.columns)), labels=corr.columns, rotation=45, ha='right', fontsize=8)
ax.set_yticks(np.arange(len(corr.columns)), labels=corr.columns, fontsize=8)
for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        ax.text(j, i, f'{corr.iloc[i, j]:.2f}', ha='center', va='center', fontsize=8)
fig.colorbar(im, ax=ax, label='Pearson correlation')
ax.set_title('Correlation Heatmap')
fig.tight_layout()
corr_path = savefig('02_correlation.png')

fig, ax = plt.subplots(figsize=(6.2, 5.1))
for cls in [0, 1]:
    part = df[df['Class'] == cls]
    ax.scatter(part['Variance'], part['Skewness'], s=12, alpha=0.65, label=f'Class {cls}')
ax.set_xlabel('Variance')
ax.set_ylabel('Skewness')
ax.set_title('Variance vs Skewness by Class')
ax.legend()
scatter_path = savefig('03_scatter.png')

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.boxplot([df[f].to_numpy() for f in features], tick_labels=features)
ax.set_ylabel('Original feature value')
ax.set_title('Feature Boxplots')
plt.xticks(rotation=15)
box_path = savefig('04_boxplots.png')

epochs = np.arange(1, 101)
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.plot(epochs, model.error_history, marker='o', markersize=1.8, linewidth=0.9)
ax.set_xlabel('Epoch')
ax.set_ylabel('Misclassified training samples')
ax.set_title('Training Error vs Epoch')
ax.grid(alpha=0.25)
error_path = savefig('05_training_error.png')

weights = np.asarray(model.weight_history)
fig, ax = plt.subplots(figsize=(8, 4.2))
for i, f in enumerate(features):
    ax.plot(epochs, weights[:, i], label=f)
ax.set_xlabel('Epoch')
ax.set_ylabel('Weight value')
ax.set_title('Weight Evolution')
ax.legend(fontsize=8)
ax.grid(alpha=0.25)
weight_path = savefig('06_weights.png')

fig, ax = plt.subplots(figsize=(8, 4.2))
ax.plot(epochs, model.bias_history, marker='o', markersize=1.8, linewidth=0.9)
ax.set_xlabel('Epoch')
ax.set_ylabel('Bias')
ax.set_title('Bias Evolution')
ax.grid(alpha=0.25)
bias_path = savefig('07_bias.png')

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5.2, 4.5))
im = ax.imshow(cm, cmap='viridis')
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=12)
ax.set_xticks([0, 1], labels=['Predicted 0', 'Predicted 1'])
ax.set_yticks([0, 1], labels=['Actual 0', 'Actual 1'])
ax.set_xlabel('Predicted class')
ax.set_ylabel('Actual class')
ax.set_title('Confusion Matrix')
fig.colorbar(im, ax=ax)
cm_path = savefig('08_confusion.png')

lr_models = {}
lr_rows = []
fig, ax = plt.subplots(figsize=(8, 4.2))
for lr in [0.001, 0.01, 0.1]:
    m = ScratchPerceptron(lr, 100).fit(X_train_scaled, y_train)
    p = m.predict(X_test_scaled)
    lr_models[lr] = m
    lr_rows.append([lr, m.error_history[-1], accuracy_score(y_test, p), precision_score(y_test, p), recall_score(y_test, p), f1_score(y_test, p)])
    ax.plot(epochs, m.error_history, label=f'learning rate = {lr}')
ax.set_xlabel('Epoch')
ax.set_ylabel('Misclassified training samples')
ax.set_title('Learning Rate Comparison')
ax.legend(fontsize=8)
ax.grid(alpha=0.25)
lr_path = savefig('09_learning_rates.png')

raw_model = ScratchPerceptron(0.01, 100).fit(X_train, y_train)
raw_pred = raw_model.predict(X_test)
std_model = model
std_pred = y_pred
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.plot(epochs, raw_model.error_history, label='Unnormalized')
ax.plot(epochs, std_model.error_history, label='Standardized')
ax.set_xlabel('Epoch')
ax.set_ylabel('Misclassified training samples')
ax.set_title('Effect of Feature Normalization on Convergence')
ax.legend(fontsize=8)
ax.grid(alpha=0.25)
norm_path = savefig('10_normalization.png')

X2 = df[['Variance', 'Skewness']].to_numpy(dtype=float)
y2 = y.copy()
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.20, random_state=42)
scaler2 = StandardScaler()
X2_train_scaled = scaler2.fit_transform(X2_train)
X2_all_scaled = scaler2.transform(X2)
model2 = ScratchPerceptron(0.01, 100).fit(X2_train_scaled, y2_train)
x_min, x_max = X2_all_scaled[:, 0].min() - 0.5, X2_all_scaled[:, 0].max() + 0.5
y_min, y_max = X2_all_scaled[:, 1].min() - 0.5, X2_all_scaled[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 320), np.linspace(y_min, y_max, 320))
grid = np.c_[xx.ravel(), yy.ravel()]
zz = model2.predict(grid).reshape(xx.shape)
fig, ax = plt.subplots(figsize=(6.2, 5.1))
ax.contourf(xx, yy, zz, alpha=0.25, levels=[-0.5, 0.5, 1.5])
for cls in [0, 1]:
    pts = X2_all_scaled[y2 == cls]
    ax.scatter(pts[:, 0], pts[:, 1], s=12, alpha=0.65, label=f'Class {cls}')
ax.set_xlabel('Standardized Variance')
ax.set_ylabel('Standardized Skewness')
ax.set_title('Two-Feature Perceptron Decision Boundary')
ax.legend(fontsize=8)
decision_path = savefig('11_decision.png')

z = np.linspace(-8, 8, 600)
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.plot(z, (z >= 0).astype(float), label='Step')
ax.plot(z, 1 / (1 + np.exp(-z)), label='Sigmoid')
ax.set_xlabel('z')
ax.set_ylabel('Activation output')
ax.set_title('Step and Sigmoid Activation Functions')
ax.legend()
ax.grid(alpha=0.25)
step_path = savefig('12_step_sigmoid.png')

logic_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)

def logic_predict(X, w, b):
    return (np.asarray(X) @ w + b >= 0).astype(int)

def logic_epoch_states(yv, learning_rate=0.2, max_epochs=20):
    w = np.zeros(2, dtype=float)
    b = 0.0
    states = [(0, w.copy(), b, int(np.sum(logic_predict(logic_X, w, b) != yv)))]
    for epoch in range(1, max_epochs + 1):
        errors = 0
        for xi, yi in zip(logic_X, yv):
            pred = 1 if np.dot(w, xi) + b >= 0 else 0
            change = learning_rate * (yi - pred)
            if change != 0:
                errors += 1
            w += change * xi
            b += change
        states.append((epoch, w.copy(), b, errors))
        if errors == 0:
            break
    return states

def logic_update_states(yv, learning_rate=0.2, updates=12):
    w = np.zeros(2, dtype=float)
    b = 0.0
    states = []
    count = 0
    while count < updates:
        for xi, yi in zip(logic_X, yv):
            pred = 1 if np.dot(w, xi) + b >= 0 else 0
            change = learning_rate * (yi - pred)
            w += change * xi
            b += change
            count += 1
            states.append((count, w.copy(), b, int(np.sum(logic_predict(logic_X, w, b) != yv))))
            if count >= updates:
                break
    return states

def draw_logic(ax, yv, w, b, title):
    gx, gy = np.meshgrid(np.linspace(-0.25, 1.25, 220), np.linspace(-0.25, 1.25, 220))
    pred = logic_predict(np.c_[gx.ravel(), gy.ravel()], w, b).reshape(gx.shape)
    ax.contourf(gx, gy, pred, alpha=0.22, levels=[-0.5, 0.5, 1.5])
    for cls in [0, 1]:
        pts = logic_X[np.asarray(yv) == cls]
        ax.scatter(pts[:, 0], pts[:, 1], s=65, edgecolors='black', linewidths=0.5)
        for px, py in pts:
            ax.text(px + 0.03, py + 0.03, str(cls), fontsize=7)
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(-0.25, 1.25)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xlabel('x1', fontsize=8)
    ax.set_ylabel('x2', fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.2)

and_y = np.array([0, 0, 0, 1])
and_states = logic_epoch_states(and_y)
fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.3))
for ax in axes.ravel():
    ax.axis('off')
for ax, (epoch, w, b, errors) in zip(axes.ravel(), and_states):
    ax.axis('on')
    title = 'AND - Initialization' if epoch == 0 else f'AND - Epoch {epoch} ({errors} update errors)'
    draw_logic(ax, and_y, w, b, title)
fig.suptitle('AND Gate - Boundary from Initialization to Convergence', fontsize=13)
fig.tight_layout()
and_path = savefig('13_and_epochs.png')

or_y = np.array([0, 1, 1, 1])
or_states = logic_epoch_states(or_y)
fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.3))
for ax in axes.ravel():
    ax.axis('off')
for ax, (epoch, w, b, errors) in zip(axes.ravel(), or_states):
    ax.axis('on')
    title = 'OR - Initialization' if epoch == 0 else f'OR - Epoch {epoch} ({errors} update errors)'
    draw_logic(ax, or_y, w, b, title)
fig.suptitle('OR Gate - Boundary from Initialization to Convergence', fontsize=13)
fig.tight_layout()
or_path = savefig('14_or_epochs.png')

xor_y = np.array([0, 1, 1, 0])
xor_states = logic_update_states(xor_y, updates=12)
fig, axes = plt.subplots(3, 4, figsize=(11.2, 8.2))
for ax, (update, w, b, errors) in zip(axes.ravel(), xor_states):
    draw_logic(ax, xor_y, w, b, f'Update {update} - {errors} misclassified')
fig.suptitle('XOR Gate - Consecutive Perceptron Updates', fontsize=13)
fig.tight_layout()
xor_path = savefig('15_xor_updates.png')

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='RTitle', parent=styles['Title'], fontName='DejaVuSerif-Bold', fontSize=15, leading=18, alignment=TA_CENTER, spaceAfter=8))
styles.add(ParagraphStyle(name='H1R', parent=styles['Heading1'], fontName='DejaVuSerif-Bold', fontSize=13, leading=15, spaceBefore=6, spaceAfter=5))
styles.add(ParagraphStyle(name='H2R', parent=styles['Heading2'], fontName='DejaVuSerif-Bold', fontSize=11, leading=13, spaceBefore=5, spaceAfter=4))
styles.add(ParagraphStyle(name='BodyR', parent=styles['BodyText'], fontName='DejaVuSerif', fontSize=9.2, leading=12, spaceAfter=5))
styles.add(ParagraphStyle(name='SmallR', parent=styles['BodyText'], fontName='DejaVuSerif', fontSize=7.5, leading=9))
styles.add(ParagraphStyle(name='CaptionR', parent=styles['BodyText'], fontName='DejaVuSerif', fontSize=8.5, leading=10, alignment=TA_CENTER, spaceAfter=5))

def page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont('DejaVuSerif', 8)
    canvas.drawCentredString(A4[0] / 2, 0.65 * cm, str(doc.page))
    canvas.restoreState()

def tstyle(font=7.4, header=True):
    cmds = [
        ('FONTNAME', (0, 0), (-1, -1), 'DejaVuSerif'),
        ('FONTSIZE', (0, 0), (-1, -1), font),
        ('LEADING', (0, 0), (-1, -1), font + 2),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('GRID', (0, 0), (-1, -1), 0.35, colors.HexColor('#666666')),
    ]
    if header:
        cmds += [('FONTNAME', (0, 0), (-1, 0), 'DejaVuSerif-Bold'), ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#EEEEEE'))]
    return TableStyle(cmds)

def add_plot(story, path, width_cm, height_cm, caption):
    story.append(Image(str(path), width=width_cm * cm, height=height_cm * cm))
    story.append(Paragraph(caption, styles['CaptionR']))

def p(text):
    return Paragraph(text, styles['BodyR'])

story = []
info = [
    ['Name', 'Joshua', 'Roll Number', '24110085'],
    ['Degree & Branch', 'B.Tech Artificial Intelligence & Data Science', 'Semester', 'V'],
    ['Subject Code & Name', 'CS3807 - Deep Learning Laboratory', 'AY', '2026-27'],
    ['Batch', '2', 'Experiment Date', '09 July 2026']
]
t = Table(info, colWidths=[3.0*cm, 7.2*cm, 2.7*cm, 3.0*cm])
t.setStyle(tstyle(7.8, header=False))
story.append(Paragraph('Shiv Nadar University Chennai', styles['RTitle']))
story.append(t)
story.append(Spacer(1, 0.2*cm))
story.append(Paragraph('Experiment 1', styles['RTitle']))
story.append(Paragraph('Implementation of a Single Layer Perceptron for Binary Classification', styles['H2R']))
story.append(Paragraph('1 Objective', styles['H1R']))
story.append(p('To build a Single Layer Perceptron from scratch and use it to classify banknotes as authentic or forged. The experiment also studies the data, training error, learning rate, normalization, and model performance.'))
story.append(Paragraph('2 Background Theory', styles['H1R']))
story.append(p('A perceptron is one artificial neuron. It multiplies each input by a weight, adds a bias, and uses an activation function.'))
story.append(p('<b>z = w<super>T</super>x + b</b><br/><b>ŷ = f(z)</b>'))
story.append(p('For a wrong prediction, the parameters are updated as follows:'))
story.append(p('<b>w<sub>new</sub> = w<sub>old</sub> + η(y - ŷ)x</b><br/><b>b<sub>new</sub> = b<sub>old</sub> + η(y - ŷ)</b>'))
story.append(p('Here, η is the learning rate. A single perceptron can learn only a linear decision boundary.'))
story.append(Paragraph('3 Activation Function', styles['H1R']))
story.append(p('The experiment uses the Step function: f(z) = 1 for z ≥ 0 and 0 for z &lt; 0. It gives a hard class output. Sigmoid gives a smooth output between 0 and 1, so it is more useful in networks trained with backpropagation.'))
act = [['Activation', 'Output', 'Common use'], ['Step', '0 or 1', 'Classical perceptron'], ['Sigmoid', '0 to 1', 'Binary output layer'], ['Tanh', '-1 to 1', 'Hidden layers'], ['ReLU', '0 to ∞', 'Modern hidden layers'], ['Leaky ReLU', '-∞ to ∞', 'Avoiding dying ReLU'], ['Softmax', 'Probabilities', 'Multi-class output']]
ta = Table(act, colWidths=[3.3*cm, 3.2*cm, 7.1*cm])
ta.setStyle(tstyle(7.6))
story.append(ta)
story.append(PageBreak())

story.append(Paragraph('4 Dataset', styles['H1R']))
story.append(p('The Banknote Authentication dataset was used. It contains 1372 rows, four numerical features, two classes, and no missing values.'))
dset = [['Item', 'Value'], ['Source', 'UCI Machine Learning Repository'], ['Features', 'Variance, Skewness, Curtosis, Entropy'], ['Class 0', 'Authentic banknote'], ['Class 1', 'Forged banknote'], ['Training rows', str(len(X_train))], ['Testing rows', str(len(X_test))]]
td = Table(dset, colWidths=[4.2*cm, 10.2*cm])
td.setStyle(tstyle(7.5))
story.append(td)
story.append(Paragraph('4.1 First Five Samples', styles['H2R']))
head = [columns] + [[f'{v:.5g}' if isinstance(v, (float, np.floating)) else str(v) for v in row] for row in df.head().to_numpy()]
th = Table(head, colWidths=[2.7*cm]*5)
th.setStyle(tstyle(6.9))
story.append(th)
story.append(Paragraph('4.2 Descriptive Statistics', styles['H2R']))
desc = df.describe().round(4)
desc_rows = [['Statistic'] + list(desc.columns)] + [[idx] + [f'{v:.4f}' for v in desc.loc[idx]] for idx in desc.index]
tdesc = Table(desc_rows, colWidths=[2.4*cm] + [2.75*cm]*5)
tdesc.setStyle(tstyle(6.2))
story.append(tdesc)
story.append(Paragraph('5 Procedure', styles['H1R']))
story.append(p('First, I checked the dataset and looked for missing values. I then made graphs to understand the features. The data was divided into training and testing sets. Only the training data was used to fit the scaler. After that, I trained the perceptron and checked its performance using the required metrics. I also tested different learning rates, raw data, standardized data, and the Scikit-learn model.'))
story.append(Paragraph('6 Source Code', styles['H1R']))
story.append(p('https://github.com/Joshua-Divide/LAB'))
story.append(PageBreak())

story.append(Paragraph('7 Generated Plots', styles['H1R']))
story.append(Paragraph('7.1 Feature Histograms', styles['H2R']))
add_plot(story, hist_path, 15.5, 10.6, 'Figure 1: Feature histograms')
story.append(p('The four features have visibly different ranges and distribution shapes. Curtosis has a pronounced right tail, while the other features show different degrees of asymmetry and spread.'))
story.append(PageBreak())

story.append(Paragraph('7.2 Correlation Heatmap', styles['H2R']))
add_plot(story, corr_path, 10.8, 9.4, 'Figure 2: Correlation heatmap')
story.append(p(f'Variance has a strong negative correlation with the class ({corr.loc["Variance", "Class"]:.2f}), while skewness is also related to the class ({corr.loc["Skewness", "Class"]:.2f}). Skewness and curtosis show a strong negative feature-to-feature relationship ({corr.loc["Skewness", "Curtosis"]:.2f}).'))
story.append(Paragraph('7.3 Scatter Plot', styles['H2R']))
add_plot(story, scatter_path, 9.0, 7.2, 'Figure 3: Variance and skewness by class')
story.append(p('Most points from the two classes occupy different regions, but there is overlap in this two-dimensional projection. Therefore, variance and skewness alone do not provide perfect linear separation.'))
story.append(PageBreak())

story.append(Paragraph('7.4 Feature Boxplots', styles['H2R']))
add_plot(story, box_path, 14.2, 7.7, 'Figure 4: Feature boxplots')
story.append(p('The boxplots show that the four features have different spreads and contain several extreme values. Standardization places their numerical scales on a comparable basis.'))
story.append(Paragraph('7.5 Training Error', styles['H2R']))
add_plot(story, error_path, 14.2, 7.4, 'Figure 5: Training error against epoch')
story.append(p(f'The number of mistakes drops sharply early in training and then fluctuates. The model ends with {model.error_history[-1]} misclassified training samples after 100 epochs.'))
story.append(PageBreak())

story.append(Paragraph('7.6 Weight Evolution', styles['H2R']))
add_plot(story, weight_path, 14.2, 7.4, 'Figure 6: Weight evolution')
story.append(p('The weights change most strongly during the earlier updates as the perceptron adjusts its linear boundary. Later changes are smaller, although the non-zero training error prevents complete stabilization.'))
story.append(Paragraph('7.7 Bias Evolution', styles['H2R']))
add_plot(story, bias_path, 14.2, 7.4, 'Figure 7: Bias evolution')
story.append(p('The bias changes along with the weights and shifts the learned decision hyperplane away from the origin.'))
story.append(PageBreak())

story.append(Paragraph('7.8 Confusion Matrix', styles['H2R']))
add_plot(story, cm_path, 9.5, 8.2, 'Figure 8: Confusion matrix')
story.append(p(f'Using forged banknotes as the positive class: TN = {cm[0,0]}, FP = {cm[0,1]}, FN = {cm[1,0]}, and TP = {cm[1,1]}.'))
story.append(Paragraph('7.9 Learning Rate Comparison', styles['H2R']))
add_plot(story, lr_path, 13.2, 7.0, 'Figure 9: Learning-rate comparison')
story.append(p('Learning rates 0.001, 0.01, and 0.1 were tested using the same initialization and training order. Their convergence curves can be compared directly from the plot.'))
story.append(PageBreak())

story.append(Paragraph('7.10 Normalization Comparison', styles['H2R']))
add_plot(story, norm_path, 13.6, 7.2, 'Figure 10: Raw and standardized training')
story.append(p(f'The raw model ends with {raw_model.error_history[-1]} training errors, while the standardized model ends with {std_model.error_history[-1]}. Standardization also changes the convergence trajectory.'))
story.append(Paragraph('7.11 Decision Boundary', styles['H2R']))
add_plot(story, decision_path, 10.0, 8.1, 'Figure 11: Two-feature decision boundary')
story.append(p('The two-feature model produces a straight decision line because a single perceptron has a linear boundary. This visualization uses only variance and skewness, so it omits information from curtosis and entropy.'))
story.append(PageBreak())

story.append(Paragraph('7.12 Logic Gate Visualization with a Single-Layer Perceptron', styles['H1R']))
story.append(Paragraph('AND Gate - Boundary from Initialization to Convergence', styles['H2R']))
add_plot(story, and_path, 16.5, 9.4, 'Figure 12: AND gate from initialization through every epoch until convergence')
story.append(p('The AND truth table is linearly separable. Starting from zero weights and zero bias, the perceptron updates its boundary and reaches an epoch with zero update errors.'))
story.append(PageBreak())

story.append(Paragraph('OR Gate - Boundary from Initialization to Convergence', styles['H2R']))
add_plot(story, or_path, 16.5, 9.4, 'Figure 13: OR gate from initialization through every epoch until convergence')
story.append(p('The OR truth table is also linearly separable. The sequence shows every epoch from initialization until the perceptron finds a separating boundary with zero update errors.'))
story.append(PageBreak())

story.append(Paragraph('XOR Gate - Consecutive Perceptron Updates', styles['H2R']))
add_plot(story, xor_path, 16.5, 12.0, 'Figure 14: Twelve consecutive XOR perceptron updates, arranged four per row')
story.append(p('XOR is not linearly separable. The boundary keeps changing as the perceptron tries to satisfy conflicting samples, so a single linear decision boundary cannot classify all four XOR points correctly.'))
story.append(PageBreak())

story.append(Paragraph('7.13 Step and Sigmoid', styles['H2R']))
add_plot(story, step_path, 13.5, 7.2, 'Figure 15: Step and Sigmoid functions')
story.append(p('The Step function changes abruptly from one class output to the other. Sigmoid changes smoothly and has a derivative, which makes gradient-based backpropagation possible.'))
story.append(Paragraph('8 Performance Tables', styles['H1R']))
story.append(Paragraph('8.1 Training Summary', styles['H2R']))
summary_rows = [
    ['Item', 'Value'], ['Dataset size', str(len(df))], ['Train/Test split', f'{len(X_train)} / {len(X_test)}'], ['Learning rate', '0.01'], ['Maximum epochs', '100'], ['Epochs run', '100'], ['Final training errors', str(model.error_history[-1])], ['Final weights', '[' + ', '.join(f'{v:.6f}' for v in model.weights) + ']'], ['Final bias', f'{model.bias:.6f}'], ['Accuracy', f'{metrics["Accuracy"]:.6f}'], ['Precision', f'{metrics["Precision"]:.6f}'], ['Recall', f'{metrics["Recall"]:.6f}'], ['F1-score', f'{metrics["F1-score"]:.6f}']
]
ts = Table(summary_rows, colWidths=[5.0*cm, 9.5*cm])
ts.setStyle(tstyle(7.2))
story.append(ts)
story.append(PageBreak())

story.append(Paragraph('8.2 First Five Epochs', styles['H2R']))
first = [['Epoch', 'Errors', 'Bias', 'w1', 'w2', 'w3', 'w4']]
for i in range(5):
    first.append([str(i+1), str(model.error_history[i]), f'{model.bias_history[i]:.4f}'] + [f'{v:.4f}' for v in model.weight_history[i]])
tf = Table(first, colWidths=[1.5*cm, 1.5*cm, 1.7*cm] + [2.15*cm]*4)
tf.setStyle(tstyle(6.7))
story.append(tf)
story.append(Paragraph('8.3 Learning Rates', styles['H2R']))
lrt = [['Rate', 'Errors', 'Accuracy', 'Precision', 'Recall', 'F1']]
for r in lr_rows:
    lrt.append([f'{r[0]:.3f}', str(r[1])] + [f'{v:.6f}' for v in r[2:]])
tl = Table(lrt, colWidths=[1.7*cm, 1.7*cm] + [2.5*cm]*4)
tl.setStyle(tstyle(6.7))
story.append(tl)
story.append(Paragraph('8.4 Scratch and Scikit-learn', styles['H2R']))
sk_model = Perceptron(max_iter=100, tol=None, eta0=0.01, fit_intercept=True, shuffle=False, random_state=42)
sk_model.fit(X_train_scaled, y_train)
sk_pred = sk_model.predict(X_test_scaled)
comp = [['Model', 'Accuracy', 'Precision', 'Recall', 'F1'], ['Scratch', f'{accuracy_score(y_test, y_pred):.6f}', f'{precision_score(y_test, y_pred):.6f}', f'{recall_score(y_test, y_pred):.6f}', f'{f1_score(y_test, y_pred):.6f}'], ['Scikit-learn', f'{accuracy_score(y_test, sk_pred):.6f}', f'{precision_score(y_test, sk_pred):.6f}', f'{recall_score(y_test, sk_pred):.6f}', f'{f1_score(y_test, sk_pred):.6f}']]
tc = Table(comp, colWidths=[3.3*cm] + [2.7*cm]*4)
tc.setStyle(tstyle(6.9))
story.append(tc)
story.append(Paragraph('8.5 Normalization Result', styles['H2R']))
norm_rows = [['Data', 'Errors', 'Accuracy', 'Precision', 'Recall', 'F1'], ['Raw', str(raw_model.error_history[-1]), f'{accuracy_score(y_test, raw_pred):.6f}', f'{precision_score(y_test, raw_pred):.6f}', f'{recall_score(y_test, raw_pred):.6f}', f'{f1_score(y_test, raw_pred):.6f}'], ['Standardized', str(std_model.error_history[-1]), f'{accuracy_score(y_test, std_pred):.6f}', f'{precision_score(y_test, std_pred):.6f}', f'{recall_score(y_test, std_pred):.6f}', f'{f1_score(y_test, std_pred):.6f}']]
tn = Table(norm_rows, colWidths=[2.7*cm, 1.6*cm] + [2.45*cm]*4)
tn.setStyle(tstyle(6.7))
story.append(tn)
story.append(Paragraph('9 Additional Questions', styles['H1R']))
story.append(Paragraph('9.1 Why is Sigmoid better for deep learning?', styles['H2R']))
story.append(p('Sigmoid changes smoothly and has a derivative. This allows gradients to be propagated through a network during backpropagation, whereas the Step function is not suitable for gradient-based training.'))
story.append(Paragraph('9.2 What is the effect of learning rate?', styles['H2R']))
story.append(p('The learning rate controls the size of each perceptron update. Smaller values create smaller parameter changes and larger values create larger changes; the plotted experiments compare their convergence behavior.'))
story.append(PageBreak())

story.append(Paragraph('9.3 Why can a single perceptron not solve XOR?', styles['H2R']))
story.append(p('XOR places the positive and negative examples on opposite corners of the input square. No single straight line can separate the two classes, so a hidden layer or another non-linear model is required.'))
story.append(Paragraph('9.4 Why does normalization help?', styles['H2R']))
story.append(p('The perceptron update uses feature values directly. Standardization makes the feature magnitudes comparable, preventing large-scale features from dominating individual parameter updates and often improving convergence behavior.'))
story.append(Paragraph('10 Discussion', styles['H1R']))
story.append(p(f'The scratch perceptron reached {metrics["Accuracy"]*100:.2f}% test accuracy in this run. It still produced {model.error_history[-1]} training errors after 100 epochs, showing that it did not reach zero training error within the chosen training duration. The confusion matrix and normalization comparison show how the learned classifier behaves on the held-out data. The AND and OR demonstrations converge because their truth tables are linearly separable, while the XOR update sequence demonstrates the limitation of a single linear perceptron.'))
story.append(Paragraph('11 Conclusion', styles['H1R']))
story.append(p(f'The Single Layer Perceptron was implemented from scratch and evaluated on the Banknote Authentication dataset. It achieved an accuracy of {metrics["Accuracy"]:.6f} and an F1-score of {metrics["F1-score"]:.6f} in this run. The experiment demonstrates the role of weights, bias, learning rate, feature scaling, and linear separability. The logic-gate visualizations reinforce that a single perceptron can learn AND and OR but cannot represent XOR.'))
story.append(Paragraph('12 References', styles['H1R']))
refs = [
    '1. F. Rosenblatt, “The Perceptron,” Psychological Review, 1958.',
    '2. UCI Machine Learning Repository, Banknote Authentication Dataset.',
    '3. Scikit-learn Documentation, Perceptron and StandardScaler.',
    '4. I. Goodfellow, Y. Bengio, and A. Courville, Deep Learning, MIT Press, 2016.'
]
for r in refs:
    story.append(p(r))

doc = SimpleDocTemplate(str(OUT), pagesize=A4, rightMargin=1.5*cm, leftMargin=1.5*cm, topMargin=1.25*cm, bottomMargin=1.25*cm, title='Experiment 1', author='Joshua')
doc.build(story, onFirstPage=page_number, onLaterPages=page_number)
