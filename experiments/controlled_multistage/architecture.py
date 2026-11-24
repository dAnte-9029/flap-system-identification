"""Generate the architecture specification for the locked model family."""
import argparse
import json
from workspace import RUN


def specification(representative):
    memory=representative=='s3a'
    nodes=[]
    def node(key,title,detail,x,y,kind='backend'):
        nodes.append(dict(id=key,type=kind,label=title,sublabel=detail,pos=[x,y],size=[202,76]))
    node('history','History and initial state','0.5 s of causal observations',35,65,'external')
    node('encoder','History encoder','64-unit GRU',290,65)
    node('commands','Future commands','Four channels, native ZOH',35,265,'external')
    node('clock','Control and phase','Causal frequency evolution',290,265)
    node('dynamics','Controlled dynamics','Control memory and MLP' if memory else 'Slow-state GRU and history field',550,65)
    node('harmonics','Periodic motion','Two harmonics, six axes',550,265)
    node('kinematics','Kinematic integration','Translation and quaternion',805,65)
    node('output','Predicted trajectory','Position, velocity, attitude, rate',1060,65,'frontend')
    edges=[('history','encoder'),('encoder','dynamics'),('commands','clock'),('clock','dynamics'),
           ('clock','harmonics'),('dynamics','harmonics'),('dynamics','kinematics'),
           ('harmonics','kinematics'),('kinematics','output')]
    return dict(schema_version=1,diagram_type='architecture',
                meta=dict(title='Control-conditioned flight dynamics',quality_profile='showcase'),
                components=nodes,connections=[dict(id=a+'-'+b,**{'from':a,'to':b}) for a,b in edges])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('out')
    args=p.parse_args()
    from pathlib import Path
    Path(args.out).write_text(json.dumps(specification(args.name),indent=2))
