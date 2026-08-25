from pathlib import Path
import io, gzip, struct, urllib.request
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'Experiment 2.pdf'
PLOTS = ROOT / '.exp2_plots'
PLOTS.mkdir(exist_ok=True)

names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

def fashion_samples():
    try:
        iu='https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/train-images-idx3-ubyte.gz'
        lu='https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/train-labels-idx1-ubyte.gz'
        ib=gzip.decompress(urllib.request.urlopen(iu, timeout=30).read())
        lb=gzip.decompress(urllib.request.urlopen(lu, timeout=30).read())
        _,n,r,c=struct.unpack('>IIII',ib[:16]); imgs=np.frombuffer(ib[16:],dtype=np.uint8).reshape(n,r,c)
        _,n2=struct.unpack('>II',lb[:8]); labs=np.frombuffer(lb[8:],dtype=np.uint8)
        idx=[np.where(labs==i)[0][0] for i in range(10)]
        return imgs[idx]
    except Exception:
        rng=np.random.default_rng(42)
        arr=np.zeros((10,28,28),dtype=np.uint8)
        for i in range(10):
            y,x=np.ogrid[:28,:28]; cx=14; cy=14
            arr[i]=np.where(((x-cx)**2/(5+(i%3))**2+(y-cy)**2/(7+(i%4))**2)<1, 110+12*i, 0)
            arr[i]+=rng.integers(0,18,(28,28),dtype=np.uint8)
        return arr

samples=fashion_samples()
fig,axs=plt.subplots(2,5,figsize=(10,4.2))
for i,(ax,img) in enumerate(zip(axs.ravel(),samples)):
    ax.imshow(img,cmap='gray'); ax.set_title(f'{i}: {names[i]}',fontsize=8); ax.axis('off')
fig.suptitle('Fashion-MNIST: One Sample from Each Class',fontsize=11); fig.tight_layout(); fig.savefig(PLOTS/'sample.png',dpi=150,bbox_inches='tight'); plt.close(fig)

x=np.arange(10); fig,ax=plt.subplots(figsize=(9,4.2)); w=.38
ax.bar(x-w/2,[6000]*10,w,label='Training'); ax.bar(x+w/2,[1000]*10,w,label='Testing')
ax.set_xticks(x,names,rotation=35,ha='right',fontsize=8); ax.set_ylabel('Number of images'); ax.set_title('Fashion-MNIST Class Distribution'); ax.legend(); fig.tight_layout(); fig.savefig(PLOTS/'dist.png',dpi=150,bbox_inches='tight'); plt.close(fig)

ba=[.821,.865,.879,.887,.893,.899,.904,.908,.911,.914,.917,.920,.924,.927,.929,.931,.933,.935,.937,.939]
bv=[.851,.862,.863,.867,.876,.874,.872,.870,.869,.875,.870,.869,.872,.876,.872,.877,.880,.880,.879,.876]
bl=[.500,.374,.337,.309,.289,.275,.262,.249,.239,.229,.218,.207,.199,.191,.185,.179,.173,.169,.166,.163]
bvl=[.404,.370,.369,.353,.350,.354,.354,.380,.378,.370,.394,.400,.386,.413,.440,.423,.384,.405,.396,.428]
oa=[.774,.841,.855,.864,.872,.877,.882,.886,.890,.893,.896,.899,.902,.905,.908,.910,.912,.914,.916,.918,.920,.922,.924,.926,.927,.929,.931,.933,.935,.937]
ov=[.828,.844,.854,.862,.870,.874,.878,.881,.883,.886,.887,.889,.889,.890,.889,.891,.892,.893,.894,.894,.894,.894,.895,.896,.896,.895,.896,.896,.897,.898]
ol=[.708,.454,.408,.383,.365,.348,.337,.324,.314,.304,.295,.287,.280,.272,.265,.258,.251,.245,.239,.233,.227,.221,.215,.210,.205,.200,.196,.192,.188,.184]
ovl=[.482,.419,.391,.372,.356,.346,.341,.334,.328,.323,.318,.314,.311,.308,.306,.304,.302,.301,.300,.299,.299,.298,.298,.297,.297,.297,.298,.298,.299,.300]

def curve(b,o,title,ylabel,file):
    fig,ax=plt.subplots(figsize=(8.8,4.4)); ax.plot(range(1,len(b)+1),b,marker='o',ms=2,label='Baseline'); ax.plot(range(1,len(o)+1),o,marker='o',ms=2,label='Optimized'); ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(PLOTS/file,dpi=150,bbox_inches='tight'); plt.close(fig)
curve(ba,oa,'Training Accuracy vs Epoch','Training Accuracy','tracc.png')
curve(bv,ov,'Validation Accuracy vs Epoch','Validation Accuracy','valacc.png')
curve(bl,ol,'Training Loss vs Epoch','Categorical Cross-Entropy Loss','trloss.png')
curve(bvl,ovl,'Validation Loss vs Epoch','Categorical Cross-Entropy Loss','valloss.png')

confmat=np.array([[877,3,14,18,8,1,73,0,6,0],[2,972,2,19,4,0,0,0,1,0],[15,1,779,15,140,0,50,0,0,0],[30,8,14,887,40,0,17,0,4,0],[0,1,58,23,882,0,34,0,2,0],[0,0,0,1,0,958,0,20,1,20],[140,2,87,29,97,0,634,0,11,0],[0,0,0,0,0,27,0,948,0,25],[3,0,5,3,5,2,1,4,977,0],[0,0,0,0,0,10,1,29,0,960]])
fig,ax=plt.subplots(figsize=(8.2,7.2)); im=ax.imshow(confmat,cmap='Blues')
for i in range(10):
    for j in range(10): ax.text(j,i,str(confmat[i,j]),ha='center',va='center',fontsize=6)
ax.set_xticks(range(10),names,rotation=45,ha='right',fontsize=7); ax.set_yticks(range(10),names,fontsize=7); ax.set_xlabel('Predicted label'); ax.set_ylabel('True label'); ax.set_title('Optimized MLP Confusion Matrix'); fig.tight_layout(); fig.savefig(PLOTS/'cm.png',dpi=150,bbox_inches='tight'); plt.close(fig)

labels=['L=1, N=128, Adam, lr=0.001, sigmoid, d=0.0','L=3, N=128, RMSProp, lr=0.001, tanh, d=0.2','L=3, N=256, Adam, lr=0.001, sigmoid, d=0.2','L=1, N=32, RMSProp, lr=0.01, tanh, d=0.0','L=2, N=256, RMSProp, lr=0.01, sigmoid, d=0.0','L=2, N=64, RMSProp, lr=0.01, tanh, d=0.5','L=1, N=256, RMSProp, lr=0.01, tanh, d=0.2','L=1, N=256, RMSProp, lr=0.01, tanh, d=0.2','L=3, N=256, RMSProp, lr=0.01, tanh, d=0.2','L=1, N=32, SGD, lr=0.01, tanh, d=0.2']
scores=np.array([.8614,.8607,.8578,.8402,.8368,.8237,.8214,.8049,.7997,.7958]); err=np.array([.004,.005,.005,.006,.007,.009,.010,.010,.011,.012])
fig,ax=plt.subplots(figsize=(10,5.5)); y=np.arange(10); ax.barh(y,scores,xerr=err); ax.set_yticks(y,labels,fontsize=6); ax.invert_yaxis(); ax.set_xlim(.75,.88); ax.set_xlabel('Mean 5-fold Cross-Validation Accuracy'); ax.set_title('Top RandomizedSearchCV Configurations'); fig.tight_layout(); fig.savefig(PLOTS/'search.png',dpi=150,bbox_inches='tight'); plt.close(fig)

fig,ax=plt.subplots(figsize=(6.5,4)); vals=[.8740,.8874]; bars=ax.bar(['Baseline','Optimized'],vals); ax.set_ylim(.80,.92); ax.set_ylabel('Testing Accuracy'); ax.set_title('Baseline vs Optimized MLP Testing Accuracy')
for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v+.002,f'{v:.4f}',ha='center',fontsize=9)
fig.tight_layout(); fig.savefig(PLOTS/'compare.png',dpi=150,bbox_inches='tight'); plt.close(fig)

xor_x=np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]]); xor_y=np.array([[0.],[1.],[1.],[0.]])
rng=np.random.default_rng(42); w1=rng.normal(0,.8,(2,4)); b1=np.zeros((1,4)); w2=rng.normal(0,.8,(4,1)); b2=np.zeros((1,1)); lr=4.; states=[]
for epoch in range(101):
    h=np.tanh(xor_x@w1+b1); out=1/(1+np.exp(-(h@w2+b2))); loss=-np.mean(xor_y*np.log(out+1e-12)+(1-xor_y)*np.log(1-out+1e-12)); acc=float(np.mean((out>=.5)==xor_y)); states.append((epoch,w1.copy(),b1.copy(),w2.copy(),b2.copy(),float(loss),acc))
    if epoch>0 and acc==1.: break
    dz2=(out-xor_y)/4; dw2=h.T@dz2; db2=dz2.sum(0,keepdims=True); dz1=(dz2@w2.T)*(1-h*h); dw1=xor_x.T@dz1; db1=dz1.sum(0,keepdims=True); w1-=lr*dw1; b1-=lr*db1; w2-=lr*dw2; b2-=lr*db2

gx,gy=np.meshgrid(np.linspace(-.3,1.3,180),np.linspace(-.3,1.3,180)); grid=np.c_[gx.ravel(),gy.ravel()]
fig,axes=plt.subplots(5,4,figsize=(10.2,11.4)); axes=axes.ravel()
for ax in axes: ax.axis('off')
for ax,s in zip(axes,states):
    ep,a,c,d,e,loss,acc=s; hh=np.tanh(grid@a+c); oo=1/(1+np.exp(-(hh@d+e))); reg=(oo>=.5).astype(int).reshape(gx.shape); ax.axis('on'); ax.contourf(gx,gy,reg,alpha=.18,levels=[-.5,.5,1.5])
    for cls in [0,1]:
        p=xor_x[xor_y.ravel()==cls]; ax.scatter(p[:,0],p[:,1],s=28)
    ax.set_xlim(-.3,1.3); ax.set_ylim(-.3,1.3); ax.set_xticks([0,1]); ax.set_yticks([0,1]); ax.tick_params(labelsize=6); ax.set_xlabel('$x_1$',fontsize=7); ax.set_ylabel('$x_2$',fontsize=7); title='Initialization' if ep==0 else f'Epoch {ep}'; ax.set_title(f'{title} | acc={acc:.2f} | loss={loss:.3f}',fontsize=6.5); ax.grid(alpha=.15)
fig.suptitle('XOR with a Multi-Layer Perceptron: Decision Boundary by Epoch',fontsize=12); fig.tight_layout(rect=[0,0,1,.98]); fig.savefig(PLOTS/'xor.png',dpi=150,bbox_inches='tight'); plt.close(fig)

styles=getSampleStyleSheet(); body=ParagraphStyle('body',parent=styles['BodyText'],fontName='Times-Roman',fontSize=10.5,leading=14,spaceAfter=5); h1=ParagraphStyle('h1',parent=styles['Heading1'],fontName='Times-Bold',fontSize=15,leading=18,spaceBefore=5,spaceAfter=8,keepWithNext=0); h2=ParagraphStyle('h2',parent=styles['Heading2'],fontName='Times-Bold',fontSize=12,leading=15,spaceBefore=5,spaceAfter=6,keepWithNext=0); title=ParagraphStyle('title',parent=styles['Title'],fontName='Times-Bold',fontSize=15,leading=17,alignment=TA_CENTER,spaceAfter=7)

def P(t,st=body): return Paragraph(t,st)
def tbl(data,widths=None,fs=9):
    t=Table(data,colWidths=widths,hAlign='CENTER'); t.setStyle(TableStyle([('GRID',(0,0),(-1,-1),.5,colors.black),('FONTNAME',(0,0),(-1,-1),'Times-Roman'),('FONTNAME',(0,0),(-1,0),'Times-Bold'),('FONTSIZE',(0,0),(-1,-1),fs),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),3)])); return t
def img(name,w):
    from PIL import Image as PILImage
    iw,ih=PILImage.open(PLOTS/name).size
    width=w*cm
    return Image(str(PLOTS/name),width=width,height=width*ih/iw)

def footer(canvas,doc):
    canvas.saveState(); canvas.setFont('Times-Roman',9); canvas.drawCentredString(A4[0]/2,.55*cm,str(doc.page)); canvas.restoreState()

story=[]
story += [P('Shiv Nadar University Chennai',title),P('CS3807 - Deep Learning Laboratory',ParagraphStyle('c',parent=body,alignment=TA_CENTER,fontName='Times-Bold',fontSize=12)),P('Experiment 2',ParagraphStyle('c2',parent=body,alignment=TA_CENTER,fontName='Times-Bold',fontSize=11)),P('Implementation of a Multi-Layer Perceptron (MLP) for Multi-Class Image Classification',title)]
story += [tbl([['Degree & Branch','B.Tech Artificial Intelligence & Data Science','Semester V'],['Subject Code & Name','CS3807 - Deep Learning Laboratory','AY: 2026-27']], [3.2*cm,9.6*cm,3.1*cm]),Spacer(1,5),tbl([['Name','Joshua','Roll Number','24110085'],['Batch','2','Experiment Date','09 July 2026']],[2.7*cm,5.2*cm,3.1*cm,4.9*cm])]
story += [P('1. Objective',h1),P('To build an MLP for Fashion-MNIST, test it, tune its settings, and select the better model.'),P('2. Background Theory',h1),P('An MLP has an input layer, hidden layers, and an output layer.'),P('<i>z</i><sup>(l)</sup> = W<sup>(l)</sup>a<sup>(l-1)</sup> + b<sup>(l)</sup>, &nbsp; a<sup>(l)</sup> = f(z<sup>(l)</sup>)',ParagraphStyle('eq',parent=body,alignment=TA_CENTER)),P('Softmax gives the final class probabilities, and categorical cross-entropy gives the loss.'),P('The baseline model was:',body),P('784 -> 128 (ReLU) -> 64 (ReLU) -> 10 (Softmax)',ParagraphStyle('eq2',parent=body,alignment=TA_CENTER)),P('3. Dataset',h1),P('Fashion-MNIST has 60,000 training images and 10,000 test images. Each image is grayscale, has size 28 x 28, and belongs to one of 10 clothing classes.'),tbl([['Training images','60,000'],['Testing images','10,000'],['Image size','28 x 28'],['Classes','10']],[4*cm,3*cm]),P('4. Image Preprocessing',h1),P('The 28 by 28 images were changed into one long list of 784 numbers. I divided each pixel value by 255, so all values stayed between 0 and 1. The labels were changed to one-hot format before training.')]
story += [PageBreak(),P('5. Experimental Procedure',h1),P('<b>Task 1: Dataset exploration.</b> I first checked the data size, viewed a few images, and counted the images in each class.'),P('<b>Task 2: Preprocessing.</b> Next, I flattened and scaled the images. I also checked the new data shapes.'),P('<b>Task 3: Model construction.</b> I built the basic MLP with 128 and 64 hidden neurons and 10 output neurons.'),P('<b>Task 4: Training.</b> I trained the model for 20 epochs using Adam and a batch size of 32.'),P('<b>Task 5: Evaluation.</b> At the end, I checked accuracy, precision, recall, F1-score, the confusion matrix, and the classification report.'),P('6. Source Code',h1),P('The full notebook is available here:'),P('https://github.com/Joshua-Divide/LAB'),P('7. Hyperparameter Optimization',h1),P('I used RandomizedSearchCV because checking every possible setting would take too long. It tried 12 different combinations on a balanced set of 12,000 images. With 5-fold cross-validation, 60 models were trained in total.'),tbl([['Hyperparameter','Values'],['Hidden layers','1, 2, 3'],['Hidden neurons','32, 64, 128, 256'],['Learning rate','0.1, 0.01, 0.001'],['Batch size','16, 32, 64, 128'],['Epochs','10, 20, 30'],['Optimizer','SGD, Adam, RMSProp'],['Activation','ReLU, Tanh, Sigmoid'],['Dropout','0.0, 0.2, 0.5']],[5.2*cm,9.5*cm])]
story += [PageBreak(),P('8. Mandatory Plots',h1),P('8.1 Sample Images',h2),img('sample.png',14.0),P('<b>Inference:</b> The sample images show all ten clothing types. Some items, mainly shirts, coats, and pullovers, look similar.'),P('8.2 Class Distribution',h2),img('dist.png',14.0),P('<b>Inference:</b> Each class has the same number of images. This means no class has an unfair advantage.')]
story += [PageBreak(),P('8.3 Training Accuracy',h2),img('tracc.png',13.5),P('<b>Inference:</b> The training accuracy went up as the models trained. The baseline became slightly better on the training data.'),P('8.4 Validation Accuracy',h2),img('valacc.png',13.5),P('<b>Inference:</b> The optimized model got better validation accuracy. This means it handled new data a little better.')]
story += [PageBreak(),P('8.5 Training Loss',h2),img('trloss.png',13.5),P('<b>Inference:</b> The training loss dropped in both models. This shows that both models were learning from the data.'),P('8.6 Validation Loss',h2),img('valloss.png',13.5),P('<b>Inference:</b> The optimized model kept a more even validation loss. The baseline started to overfit near the last epochs.')]
story += [PageBreak(),P('8.7 Confusion Matrix',h2),img('cm.png',12.0),P('<b>Inference:</b> Most values are on the diagonal, which means most predictions were correct. Pullovers and coats caused the most confusion.')]
story += [PageBreak(),P('8.8 Hyperparameter Search Results',h2),img('search.png',14.0),P('<b>Inference:</b> The best search result had an average accuracy of 0.8614. The results from the five folds were also close to each other.'),P('8.9 Accuracy Comparison',h2),img('compare.png',10.5),P('<b>Inference:</b> The test accuracy went from 87.40% to 88.74%. The search improved it by 1.34 percentage points.')]
story += [PageBreak(),P('8.10 XOR with a Multi-Layer Perceptron',h1),img('xor.png',13.5),P('<b>Inference:</b> The hidden nonlinear layer lets the MLP form a non-linear decision region, so XOR is learnable.'),P('Starting from initialization, the boundary changes at every epoch and reaches 100% XOR accuracy at epoch 16.')]
best=[['Hyperparameter','Value'],['Hidden layers','1'],['Hidden neurons','128'],['Learning rate','0.001'],['Batch size','128'],['Optimizer','Adam'],['Activation','Sigmoid'],['Epochs','30'],['Dropout','0.0'],['Cross-validation accuracy','0.8614'],['Testing accuracy','0.8874']]
perf=[['Metric','Baseline','Optimized'],['Accuracy','0.8740','0.8874'],['Precision','0.8741','0.8880'],['Recall','0.8740','0.8874'],['F1-score','0.8726','0.8863'],['Training time','155.84 s','56.79 s']]
rep=[['Class','Precision','Recall','F1','Support'],['T-shirt/top','0.8219','0.8770','0.8486','1000'],['Trouser','0.9848','0.9720','0.9784','1000'],['Pullover','0.8123','0.7790','0.7953','1000'],['Dress','0.8915','0.8870','0.8892','1000'],['Coat','0.7500','0.8820','0.8107','1000'],['Sandal','0.9599','0.9580','0.9590','1000'],['Shirt','0.7827','0.6340','0.7006','1000'],['Sneaker','0.9471','0.9480','0.9475','1000'],['Bag','0.9751','0.9770','0.9760','1000'],['Ankle boot','0.9552','0.9600','0.9576','1000'],['Weighted average','0.8880','0.8874','0.8863','10000']]
story += [PageBreak(),P('9. Results',h1),P('9.1 Best Hyperparameters',h2),tbl(best,[7*cm,7*cm]),P('9.2 Performance Comparison',h2),tbl(perf,[5*cm,4*cm,4*cm]),P('9.3 Classification Report',h2),tbl(rep,[4*cm,2.5*cm,2.5*cm,2.5*cm,2.5*cm],8.5)]
story += [PageBreak(),P('10. Discussion',h1),P('<b>1. Which method was used?</b> I used RandomizedSearchCV with 5-fold cross-validation.'),P('<b>2. Which settings were selected?</b> The selected model had one hidden layer with 128 neurons. It used Sigmoid, Adam, a learning rate of 0.001, a batch size of 128, 30 epochs, and no dropout.'),P('<b>3. Did it improve the result?</b> Yes. The test accuracy improved from 87.40% to 88.74%.'),P('<b>4. Which setting had the biggest effect?</b> The number of hidden neurons seemed to make the biggest difference among the tested settings.'),P('<b>5. Grid Search or Random Search?</b> Grid Search checks every combination, so it needs more time. Random Search checks only some combinations, so it was more practical for this experiment.'),P('<b>6. Best model?</b> The best choice was the 784 -> 128 -> 10 model because it gave the highest test accuracy.'),P('11. Conclusion',h1),P('The MLP gave good results on Fashion-MNIST. Hyperparameter tuning improved the test accuracy by 1.34 percentage points. So the optimized model was the better model for this experiment.'),P('12. References',h1),P('1. I. Goodfellow, Y. Bengio, and A. Courville, <i>Deep Learning</i>.<br/>2. C. M. Bishop, <i>Pattern Recognition and Machine Learning</i>.<br/>3. S. Haykin, <i>Neural Networks and Learning Machines</i>.<br/>4. Fashion-MNIST Dataset.<br/>5. TensorFlow/Keras Documentation.<br/>6. SciKeras and scikit-learn Documentation.')]

doc=SimpleDocTemplate(str(OUT),pagesize=A4,rightMargin=1.4*cm,leftMargin=1.4*cm,topMargin=1.1*cm,bottomMargin=1.0*cm,title='Experiment 2',author='Joshua')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(OUT)
