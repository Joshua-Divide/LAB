import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'Experiment 3.pdf'
FIG = Path('/tmp/experiment3_figures')
FIG.mkdir(parents=True, exist_ok=True)
keras.utils.set_random_seed(42)
class_names = ['Airplane','Automobile','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck']

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
y_train = y_train.ravel()
y_test = y_test.ravel()
x_train_n = x_train.astype('float32') / 255.0
x_test_n = x_test.astype('float32') / 255.0

indices = [np.where(y_train == c)[0][0] for c in range(10)]
fig, axes = plt.subplots(2, 5, figsize=(11, 5))
for ax, idx in zip(axes.ravel(), indices):
    ax.imshow(x_train[idx])
    ax.set_title(class_names[y_train[idx]])
    ax.axis('off')
fig.tight_layout()
fig.savefig(FIG / 'sample_images.png', dpi=180, bbox_inches='tight')
plt.close(fig)

train_counts = np.bincount(y_train, minlength=10)
test_counts = np.bincount(y_test, minlength=10)
pos = np.arange(10)
width = 0.4
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(pos - width / 2, train_counts, width, label='Training')
ax.bar(pos + width / 2, test_counts, width, label='Testing')
ax.set_xticks(pos, class_names, rotation=35, ha='right')
ax.set_ylabel('Images')
ax.set_title('CIFAR-10 Class Distribution')
ax.legend()
fig.tight_layout()
fig.savefig(FIG / 'class_distribution.png', dpi=180, bbox_inches='tight')
plt.close(fig)

def output_size(n, f, s=1, p=0):
    return (n - f + 2 * p) // s + 1

def build_cnn(pooling='max', first_filters=16):
    pool = layers.MaxPooling2D if pooling == 'max' else layers.AveragePooling2D
    model = keras.Sequential([
        keras.Input(shape=(32, 32, 3)),
        layers.Conv2D(first_filters, (3, 3), padding='same', activation='relu', name='conv1'),
        pool((2, 2)),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu', name='conv2'),
        pool((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_cnn()
start = time.perf_counter()
history = model.fit(x_train_n, y_train, validation_split=0.1, epochs=20, batch_size=32, verbose=2)
training_time = time.perf_counter() - start
probabilities = model.predict(x_test_n, verbose=0)
predictions = np.argmax(probabilities, axis=1)
metrics = {
    'accuracy': accuracy_score(y_test, predictions),
    'precision': precision_score(y_test, predictions, average='weighted', zero_division=0),
    'recall': recall_score(y_test, predictions, average='weighted', zero_division=0),
    'f1': f1_score(y_test, predictions, average='weighted', zero_division=0)
}
report = classification_report(y_test, predictions, target_names=class_names, output_dict=True, zero_division=0)
cm = confusion_matrix(y_test, predictions)

def curve(key, title, ylabel, name):
    values = history.history[key]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(values) + 1), values, marker='o', markersize=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=180, bbox_inches='tight')
    plt.close(fig)

curve('accuracy', 'Training Accuracy vs Epoch', 'Accuracy', 'training_accuracy.png')
curve('val_accuracy', 'Validation Accuracy vs Epoch', 'Accuracy', 'validation_accuracy.png')
curve('loss', 'Training Loss vs Epoch', 'Loss', 'training_loss.png')
curve('val_loss', 'Validation Loss vs Epoch', 'Loss', 'validation_loss.png')

feature_model = keras.Model(model.input, model.get_layer('conv1').output)
feature_maps = feature_model.predict(x_test_n[:1], verbose=0)[0]
fig, axes = plt.subplots(2, 4, figsize=(10, 5))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(feature_maps[:, :, i], cmap='gray')
    ax.set_title(f'Feature Map {i + 1}')
    ax.axis('off')
fig.tight_layout()
fig.savefig(FIG / 'feature_maps.png', dpi=180, bbox_inches='tight')
plt.close(fig)

fig, ax = plt.subplots(figsize=(8.5, 7.5))
ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, cmap='Blues', values_format='d', colorbar=False)
plt.xticks(rotation=40, ha='right')
ax.set_title('CNN Confusion Matrix')
fig.tight_layout()
fig.savefig(FIG / 'confusion_matrix.png', dpi=180, bbox_inches='tight')
plt.close(fig)

comparison_indices = np.concatenate([np.where(y_train == c)[0][:1000] for c in range(10)])
x_compare = x_train_n[comparison_indices]
y_compare = y_train[comparison_indices]

def train_variant(pooling='max', first_filters=16):
    variant = build_cnn(pooling=pooling, first_filters=first_filters)
    start = time.perf_counter()
    variant.fit(x_compare, y_compare, epochs=3, batch_size=64, verbose=0)
    elapsed = time.perf_counter() - start
    pred = np.argmax(variant.predict(x_test_n, verbose=0), axis=1)
    return variant, accuracy_score(y_test, pred), elapsed

max_model, max_accuracy, max_time = train_variant('max', 16)
avg_model, avg_accuracy, avg_time = train_variant('avg', 16)
filter64_model, filter64_accuracy, filter64_time = train_variant('max', 64)

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyX', parent=styles['BodyText'], fontName='Times-Roman', fontSize=10.5, leading=14, spaceAfter=6))
styles.add(ParagraphStyle(name='H1X', parent=styles['Heading1'], fontName='Times-Bold', fontSize=17, leading=20, spaceBefore=10, spaceAfter=10))
styles.add(ParagraphStyle(name='H2X', parent=styles['Heading2'], fontName='Times-Bold', fontSize=13, leading=16, spaceBefore=8, spaceAfter=7))
styles.add(ParagraphStyle(name='CaptionX', parent=styles['BodyText'], fontName='Times-Roman', fontSize=9.5, leading=12, alignment=TA_CENTER, spaceAfter=7))
styles.add(ParagraphStyle(name='InferenceX', parent=styles['BodyText'], fontName='Times-Roman', fontSize=9.5, leading=12, leftIndent=0, spaceAfter=9))
styles.add(ParagraphStyle(name='TitleX', parent=styles['Title'], fontName='Times-Bold', fontSize=16, leading=18, alignment=TA_CENTER, spaceAfter=3))
styles.add(ParagraphStyle(name='TopX', parent=styles['BodyText'], fontName='Times-Bold', fontSize=14, leading=16, alignment=TA_CENTER, spaceAfter=2))
styles.add(ParagraphStyle(name='SmallX', parent=styles['BodyText'], fontName='Times-Roman', fontSize=8.7, leading=11))

body = styles['BodyX']; h1 = styles['H1X']; h2 = styles['H2X']; cap = styles['CaptionX']; inf = styles['InferenceX']; small = styles['SmallX']

def P(txt, style=body):
    return Paragraph(txt, style)

def tbl(data, widths=None, font=9.5, header=True):
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign='CENTER')
    cmds = [('GRID',(0,0),(-1,-1),0.55,colors.black),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),('FONTNAME',(0,0),(-1,-1),'Times-Roman'),('FONTSIZE',(0,0),(-1,-1),font)]
    if header:
        cmds += [('FONTNAME',(0,0),(-1,0),'Times-Bold'),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#f4f4f4'))]
    t.setStyle(TableStyle(cmds))
    return t

def fig_block(section, path, caption, inference, width=155*mm):
    im = Image(str(path))
    im.drawWidth = width
    im.drawHeight = width * im.imageHeight / im.imageWidth
    return [P(section, h2), im, P(caption, cap), P('<b>Inference:</b> ' + inference, inf)]

def page_num(canvas, doc):
    canvas.saveState()
    canvas.setFont('Times-Roman', 9)
    canvas.drawCentredString(A4[0]/2, 12*mm, str(doc.page))
    canvas.restoreState()

doc = SimpleDocTemplate(str(OUT), pagesize=A4, rightMargin=20*mm, leftMargin=20*mm, topMargin=16*mm, bottomMargin=20*mm)
story = []
story += [P('Shiv Nadar University Chennai', styles['TopX']), P('CS3807 - Deep Learning Laboratory', styles['TopX']), P('Experiment 3', styles['TopX']), P('Implementation of Convolutional Neural Networks (CNNs) for Image Classification', styles['TitleX']), Spacer(1, 4*mm)]
meta1 = [['Degree & Branch','B.Tech Artificial Intelligence & Data Science','Semester V'],['Subject Code & Name','CS3807 - Deep Learning Laboratory','AY: 2026-27']]
story += [tbl(meta1,[46*mm,93*mm,31*mm],9.5,False), Spacer(1,1.5*mm)]
meta2 = [['Name','Joshua','Roll Number','24110085'],['Batch','2','Experiment Date','']]
story += [tbl(meta2,[34*mm,55*mm,34*mm,47*mm],9.5,False), Spacer(1,5*mm)]
story += [P('1. Objective', h1), P('To understand convolution, pooling, feature-map visualization, and CIFAR-10 image classification using TensorFlow/Keras.')]
story += [P('2. Background Theory', h1), P('A CNN learns local spatial patterns using shared convolution kernels, applies non-linearity with ReLU, reduces spatial size using pooling, and performs classification using dense layers.'), P('<b>Convolution:</b> Y(i,j) = sum_m sum_n X(i+m,j+n)K(m,n)'), P('<b>Output size:</b> floor((N - F + 2P) / S) + 1'), P('<b>Model:</b> Input -> Conv(16, 3x3, Same) -> ReLU -> MaxPool -> Conv(32, 3x3, Same) -> ReLU -> MaxPool -> Flatten -> Dense(64) -> Softmax(10)')]
story += [P('2.1 Common Convolution Kernels', h2)]
kernels = [['Kernel','Size','Purpose'],['Identity','3 x 3','Preserve image'],['Sobel-X','3 x 3','Vertical edges'],['Sobel-Y','3 x 3','Horizontal edges'],['Laplacian','3 x 3','Edges in all directions'],['Sharpen','3 x 3','Enhance details'],['Box Blur','3 x 3','Smooth image'],['Gaussian Blur','3 x 3','Noise reduction'],['Emboss','3 x 3','Directional texture'],['Outline','3 x 3','Boundary extraction'],['Motion Blur','5 x 5','Simulate motion']]
story += [tbl(kernels,[42*mm,28*mm,90*mm],9), P('<b>Inference:</b> Different kernels emphasize different local image structures.', inf)]
story += [P('3. Dataset', h1), P('CIFAR-10 contains 60,000 RGB images of size 32 x 32, split into 50,000 training images and 10,000 testing images across ten balanced classes.'), tbl([['Item','Value'],['Training images','50,000'],['Testing images','10,000'],['Image size','32 x 32 x 3'],['Classes','10']],[70*mm,45*mm],9.5), P('<b>Inference:</b> The dataset is balanced across all ten classes.', inf)]
story += [P('4. Experimental Procedure', h1), P('<b>Task 1:</b> Load CIFAR-10, display ten samples, print dimensions, and plot class distribution.'), P('<b>Task 2:</b> Compare 3 x 3, 5 x 5, and 7 x 7 convolution kernels and record feature-map sizes.'), P('<b>Task 3:</b> Compare stride 1/2 and Same/Valid padding and compute output dimensions.'), P('<b>Task 4:</b> Visualize at least eight feature maps from the first convolution layer.'), P('<b>Task 5:</b> Compare max pooling and average pooling using the same controlled setup.'), P('<b>Task 6:</b> Train the required CNN for 20 epochs using Adam and batch size 32.'), P('<b>Task 7:</b> Evaluate accuracy, precision, recall, F1-score, confusion matrix, and classification report.')]
story += [P('5. Source Code', h1), P('https://github.com/Joshua-Divide/LAB')]
story += [PageBreak()]

story += [P('6. Numerical and Dimension Calculations', h1), P('6.1 Numerical Example 1: Convolution', h2), P('Input X = [[1,2,3],[4,5,6],[7,8,9]], kernel K = [[1,0],[0,1]].'), tbl([['Position','Calculation','Output'],['1','1(1)+2(0)+4(0)+5(1)','6'],['2','2(1)+3(0)+5(0)+6(1)','8'],['3','4(1)+5(0)+7(0)+8(1)','12'],['4','5(1)+6(0)+8(0)+9(1)','14']],[28*mm,95*mm,25*mm],9), P('<b>Feature map:</b> [[6, 8], [12, 14]]')]
story += [P('6.2 Numerical Example 2: Max Pooling', h2), tbl([['2 x 2 window','Maximum'],['[[1,5],[7,8]]','8'],['[[2,3],[1,0]]','3'],['[[4,6],[2,3]]','6'],['[[9,5],[1,8]]','9']],[85*mm,45*mm],9.5), P('<b>Output:</b> [[8, 3], [6, 9]]')]
story += [P('6.3 Numerical Example 3: Parameter Calculation', h2), P('For a 32 x 32 x 3 input with 16 filters of size 3 x 3: (3 x 3 x 3 + 1) x 16 = <b>448 trainable parameters</b>.')]
dims = [['Kernel / Setting','Output size'],['3 x 3, Valid, stride 1','30 x 30'],['5 x 5, Valid, stride 1','28 x 28'],['7 x 7, Valid, stride 1','26 x 26'],['3 x 3, Same, stride 1','32 x 32'],['3 x 3, Valid, stride 2','15 x 15'],['3 x 3, Same, stride 2','16 x 16']]
story += [P('6.4 Kernel, Stride and Padding Comparison', h2), tbl(dims,[95*mm,45*mm],9.5), P('<b>Inference:</b> Larger kernels or strides reduce spatial size; Same padding preserves it.', inf)]
story += [PageBreak()]

story += fig_block('7.1 Sample Images', FIG/'sample_images.png', 'Figure 1: One CIFAR-10 sample from each class.', 'One representative image is shown for every class.', 155*mm)
story += fig_block('7.2 Class Distribution', FIG/'class_distribution.png', 'Figure 2: CIFAR-10 training and testing class counts.', 'Every class has the same number of samples.', 155*mm)
story += [PageBreak()]
story += fig_block('7.3 Training Accuracy', FIG/'training_accuracy.png', 'Figure 3: Training accuracy across 20 epochs.', 'Training accuracy rises as the CNN learns.', 145*mm)
story += fig_block('7.4 Validation Accuracy', FIG/'validation_accuracy.png', 'Figure 4: Validation accuracy across 20 epochs.', 'Validation accuracy improves and then stabilizes.', 145*mm)
story += [PageBreak()]
story += fig_block('7.5 Training Loss', FIG/'training_loss.png', 'Figure 5: Training loss across 20 epochs.', 'Training loss decreases through training.', 145*mm)
story += fig_block('7.6 Validation Loss', FIG/'validation_loss.png', 'Figure 6: Validation loss across 20 epochs.', 'Validation loss tracks generalization during training.', 145*mm)
story += [PageBreak()]
story += fig_block('7.7 Feature Maps', FIG/'feature_maps.png', 'Figure 7: Eight feature maps from the first convolution layer.', 'Different filters respond to different local patterns.', 155*mm)
story += [PageBreak()]
story += fig_block('7.8 Confusion Matrix', FIG/'confusion_matrix.png', 'Figure 8: Test-set confusion matrix.', 'Most correct predictions appear on the main diagonal.', 150*mm)
story += [PageBreak()]

story += [P('8. Pooling and Filter Comparisons', h1), P('8.1 Max Pooling vs Average Pooling', h2)]
pool_data = [['Pooling','First pooled output','Test accuracy','Train time (s)'],['Max Pooling','16 x 16 x 16',f'{max_accuracy:.4f}',f'{max_time:.2f}'],['Average Pooling','16 x 16 x 16',f'{avg_accuracy:.4f}',f'{avg_time:.2f}']]
story += [tbl(pool_data,[45*mm,48*mm,35*mm,35*mm],8.8), P('<b>Inference:</b> Both preserve output size; accuracy differs by pooling operation.', inf), P('Controlled comparison: 3 epochs on a balanced 10,000-image training subset.', small)]
story += [P('8.2 First-Layer Filters: 16 vs 64', h2)]
filter_data = [['Filters','Parameters','Test accuracy','Train time (s)'],['16',f'{max_model.count_params():,}',f'{max_accuracy:.4f}',f'{max_time:.2f}'],['64',f'{filter64_model.count_params():,}',f'{filter64_accuracy:.4f}',f'{filter64_time:.2f}']]
story += [tbl(filter_data,[35*mm,42*mm,42*mm,42*mm],9), P('<b>Inference:</b> More filters increase model capacity and computation.', inf), P('Controlled comparison: 3 epochs on the same balanced 10,000-image subset.', small)]
story += [P('9. Results', h1)]
result_data = [['Metric','Value'],['Final training accuracy',f"{history.history['accuracy'][-1]:.4f}"],['Testing accuracy',f"{metrics['accuracy']:.4f}"],['Weighted precision',f"{metrics['precision']:.4f}"],['Weighted recall',f"{metrics['recall']:.4f}"],['Weighted F1-score',f"{metrics['f1']:.4f}"],['Trainable parameters',f'{model.count_params():,}'],['Training time',f'{training_time:.2f} s']]
story += [tbl(result_data,[92*mm,58*mm],9.5), P('<b>Inference:</b> The final metrics summarize CNN performance on CIFAR-10.', inf)]
story += [P('9.1 Classification Report', h2)]
class_data = [['Class','Precision','Recall','F1','Support']]
for name in class_names:
    r = report[name]
    class_data.append([name,f"{r['precision']:.4f}",f"{r['recall']:.4f}",f"{r['f1-score']:.4f}",str(int(r['support']))])
r = report['weighted avg']
class_data.append(['Weighted avg',f"{r['precision']:.4f}",f"{r['recall']:.4f}",f"{r['f1-score']:.4f}",str(int(r['support']))])
story += [tbl(class_data,[40*mm,30*mm,30*mm,30*mm,28*mm],8.2), P('<b>Inference:</b> Per-class scores show which CIFAR-10 categories are harder.', inf)]
story += [PageBreak()]

story += [P('10. Additional Exercises', h1), P('1. For N=64, F=5, S=2, P=2: floor((64 - 5 + 4)/2) + 1 = <b>32</b>. Output = <b>32 x 32</b>.'), P('2. For 64 filters of size 3 x 3 with RGB input: (3 x 3 x 3 + 1) x 64 = <b>1,792 parameters</b>.'), P('3. ReLU vs Sigmoid', h2), tbl([['Activation','Main behavior'],['ReLU','Fast, sparse activations; reduces saturation for positive values.'],['Sigmoid','Maps to 0-1 but may saturate and produce small gradients.']],[42*mm,118*mm],9), P('<b>Inference:</b> ReLU is generally better suited to hidden CNN layers.', inf), P('4. Max pooling vs average pooling is reported in Section 8.1 using an identical controlled setup.'), P('5. Increasing the first convolution layer from 16 to 64 filters is reported in Section 8.2 with parameter count, accuracy, and training time.')]
story += [P('11. Discussion', h1), P('<b>1. Why convolution instead of fully connected layers?</b> Local receptive fields and weight sharing preserve spatial structure while using far fewer parameters.'), P('<b>2. How does stride affect feature-map size?</b> Larger stride evaluates fewer positions and reduces spatial dimensions.'), P('<b>3. What is the role of padding?</b> Padding controls border handling; Same preserves size while Valid performs no padding.'), P('<b>4. Why is pooling used?</b> Pooling reduces spatial resolution, computation, and sensitivity to small translations.'), P('<b>5. How do feature maps represent characteristics?</b> Each channel records where a learned filter responds strongly to local patterns.'), P('<b>6. Why fewer parameters than an MLP?</b> CNN kernels are reused across image locations instead of learning a separate weight for every pixel connection.')]
story += [P('12. Conclusion', h1), P('The experiment implements convolution, stride, padding, pooling, feature-map visualization, and a CNN classifier on CIFAR-10. The required CNN was trained for 20 epochs with Adam and batch size 32, then evaluated using the requested classification metrics.')]
story += [P('13. References', h1), P('1. I. Goodfellow, Y. Bengio, and A. Courville, <i>Deep Learning</i>.<br/>2. C. M. Bishop, <i>Pattern Recognition and Machine Learning</i>.<br/>3. S. Haykin, <i>Neural Networks and Learning Machines</i>.<br/>4. TensorFlow/Keras Documentation.<br/>5. CIFAR-10 Dataset Documentation.')]

doc.build(story, onFirstPage=page_num, onLaterPages=page_num)
print('WROTE', OUT)
print('RESULTS', metrics, 'params', model.count_params(), 'train_time', training_time)
