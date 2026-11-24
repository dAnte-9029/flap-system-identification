"""Vector diagram of the selected forward pass; coordinates are in paper inches."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from workspace import RUN


def draw(rep):
    fig,ax=plt.subplots(figsize=(7.16,2.75))
    fig.subplots_adjust(left=.012,right=.988,top=.98,bottom=.03)
    ax.set(xlim=(-1,101),ylim=(0,42))
    ax.axis('off')
    plt.rcParams['pdf.fonttype']=42
    colors={'input':('#f5f6f8','#687482'),'slow':('#e9f3f5','#226573'),
            'fast':('#fbf1e4','#966632'),'out':('#edf1f9','#42567d')}
    def node(x,y,w,h,name,detail,kind):
        fill,line=colors[kind]
        rect=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.13,rounding_size=1.2',linewidth=.95,
                            edgecolor=line,facecolor=fill,zorder=3)
        ax.add_patch(rect)
        ax.text(x+w/2,y+h*.71,name,ha='center',va='center',fontsize=8.1,weight='semibold',zorder=4)
        ax.text(x+w/2,y+h*.32,detail,ha='center',va='center',fontsize=7.15,linespacing=1.15,zorder=4)
    def arrow(points,col='#485765'):
        x,y=zip(*points)
        ax.plot(x[:-1],y[:-1],lw=.95,color=col,solid_capstyle='round',solid_joinstyle='round',zorder=1)
        ax.annotate('',xy=points[-1],xytext=points[-2],arrowprops=dict(arrowstyle='-|>',color=col,lw=.95,
                    shrinkA=0,shrinkB=0,mutation_scale=8),zorder=2)
    node(0,25,18,12,'Flight history','0.5 s state + control\ncurrent state at T','input')
    node(0,4,18,12,'Control plan','Motor + three servos\nnative event times','input')
    node(22,25,18,12,'GRU encoder','26 history samples\n64-D encoding','slow')
    node(22,4,18,12,'Phase clock','Motor and initial f(T)\nphase integration','fast')
    memory=rep=='s3a'
    if memory:
        slow='Control-memory MLP'; detail='Three filtered controls\n20 ms target update'
    else:
        slow='Slow dynamics'; detail='48-D GRU + history MLP\n100 ms target update'
    node(46,25,24,12,slow,detail,'slow')
    node(46,4,24,12,'Fast motion','Event-driven amplitudes\ntwo harmonics per axis','fast')
    node(76,14.5,24,12,'Trajectory rollout','From current state at T\nposition, velocity, q, rate','out')
    arrow([(18,31),(22,31)])
    arrow([(18,10),(22,10)])
    arrow([(40,31),(46,31)])
    arrow([(40,10),(46,10)],'#966632')
    arrow([(18,13),(20,13),(20,21),(49,21),(49,25)])
    arrow([(39,16),(39,19),(53,19),(53,25)],'#966632')
    arrow([(64,25),(64,16)],'#226573')
    arrow([(70,31),(73,31),(73,23),(76,23)],'#226573')
    arrow([(70,10),(73,10),(73,18),(76,18)],'#966632')
    if not memory:
        arrow([(68,37),(68,40.5),(51,40.5),(51,37)],'#226573')
        ax.text(59.5,40.8,'predicted state feedback',ha='center',va='bottom',fontsize=6.4,color='#226573')
    fig.savefig(RUN/'network.pdf',bbox_inches='tight',pad_inches=.05)
    fig.savefig(RUN/'network.png',dpi=225,bbox_inches='tight',pad_inches=.05)
    plt.close(fig)


if __name__=='__main__': draw(json.loads((RUN/'freeze.json').read_text())['representative'])
