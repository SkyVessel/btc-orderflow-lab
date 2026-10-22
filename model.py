"""Small deterministic neural networks, trained locally with NumPy/Adam."""
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -40, 40)))


class MLP:
    def __init__(self, n, outputs, seed, binary):
        rng = np.random.default_rng(seed)
        sizes = [n, 32, 16, outputs]
        self.p = {}
        for i, (a,b) in enumerate(zip(sizes[:-1],sizes[1:])):
            self.p[f"w{i}"] = rng.normal(0, np.sqrt(1/a), (a,b))
            self.p[f"b{i}"] = np.zeros(b)
        self.binary = binary
        self.mean, self.scale = np.zeros(n), np.ones(n)

    def forward(self, x):
        z = np.clip((x-self.mean)/self.scale,-5,5)
        h1 = np.tanh(z@self.p["w0"]+self.p["b0"])
        h2 = np.tanh(h1@self.p["w1"]+self.p["b1"])
        out = h2@self.p["w2"]+self.p["b2"]
        if self.binary:
            out = sigmoid(out)
        return z,h1,h2,out

    def predict(self,x):
        return self.forward(x)[-1]

    def fit(self,x,y,epochs,lr,batch,l2,seed,reset_scaler=False):
        assert len(x)>0 and np.isfinite(x).all() and np.isfinite(y).all()
        y = np.asarray(y).reshape(len(x),-1)
        if reset_scaler:
            self.mean,self.scale=x.mean(axis=0),x.std(axis=0)
            self.scale=np.where(self.scale<1e-8,1,self.scale)
        m={k:np.zeros_like(v) for k,v in self.p.items()}
        v={k:np.zeros_like(v) for k,v in self.p.items()}
        rng=np.random.default_rng(seed); step=0; history=[]
        for epoch in range(epochs):
            ids=rng.permutation(len(x))
            for start in range(0,len(x),batch):
                idx=ids[start:start+batch];z,h1,h2,out=self.forward(x[idx])
                if self.binary:
                    delta=(out-y[idx])/len(idx)
                else:
                    # Huber derivative limits sensitivity to a few large return labels.
                    delta=np.clip(out-y[idx],-1,1)/(len(idx)*y.shape[1])
                g={"w2":h2.T@delta+l2*self.p["w2"],"b2":delta.sum(axis=0)}
                d2=(delta@self.p["w2"].T)*(1-h2*h2)
                g.update(w1=h1.T@d2+l2*self.p["w1"],b1=d2.sum(axis=0))
                d1=(d2@self.p["w1"].T)*(1-h1*h1)
                g.update(w0=z.T@d1+l2*self.p["w0"],b0=d1.sum(axis=0))
                step+=1
                for k in self.p:
                    m[k]=.9*m[k]+.1*g[k];v[k]=.999*v[k]+.001*g[k]*g[k]
                    self.p[k]-=lr*(m[k]/(1-.9**step))/(np.sqrt(v[k]/(1-.999**step))+1e-8)
            pred=self.predict(x)
            if self.binary:
                loss=-np.mean(y*np.log(pred+1e-12)+(1-y)*np.log(1-pred+1e-12))
            else:
                d=np.abs(pred-y);loss=np.mean(np.where(d<=1,.5*d*d,d-.5))
            history.append(float(loss))
        assert all(np.isfinite(p).all() for p in self.p.values())
        return history

    def save(self,path):
        # float64 retains deterministic replay precision; size report states actual dtype.
        np.savez_compressed(path,**self.p,mean=self.mean,scale=self.scale,binary=self.binary)

    @classmethod
    def load(cls,path):
        d=np.load(path);obj=cls(len(d["mean"]),len(d["b2"]),0,bool(d["binary"]))
        obj.p={k:d[k] for k in obj.p};obj.mean=d["mean"];obj.scale=d["scale"]
        return obj

    @property
    def parameters(self):
        return sum(p.size for p in self.p.values())
