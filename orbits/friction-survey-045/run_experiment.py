"""friction-survey-045: Survey bounded friction functions g(xi).

Key result: tanh has omega_max = 1.0 (exact), vs log-osc's 0.732.
Soft-clip and logosc(6,1) appear higher but are quadrature artifacts at small Q.
"""

import json, os, sys, time
import numpy as np
from scipy import integrate, optimize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from research.eval.potentials import GaussianMixture2D

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ===========================================================================
# Friction candidates
# ===========================================================================
class FrictionFunc:
    def __init__(self, name, g, g_prime, V, color=None, gprime_exact=None):
        self.name = name; self.g = g; self.g_prime = g_prime; self.V = V
        self.label = name; self.color = color; self.gprime_exact = gprime_exact

def make_all():
    C = []
    C.append(FrictionFunc("log-osc",
        g=lambda xi: 2*xi/(1+xi**2),
        g_prime=lambda xi: 2*(1-xi**2)/(1+xi**2)**2,
        V=lambda xi: np.log(1+xi**2), color="#1f77b4",
        gprime_exact=lambda Q: max((2*Q-1)/(Q+1), 0) if Q > 0.5 else 0))
    C.append(FrictionFunc("tanh",
        g=lambda xi: np.tanh(xi),
        g_prime=lambda xi: 1.0/np.cosh(xi)**2,
        V=lambda xi: np.log(np.cosh(xi)), color="#ff7f0e",
        gprime_exact=lambda Q: Q/(Q+1)))
    C.append(FrictionFunc("arctan",
        g=lambda xi: (2/np.pi)*np.arctan(xi),
        g_prime=lambda xi: (2/np.pi)/(1+xi**2),
        V=lambda xi: (2/np.pi)*(xi*np.arctan(xi)-0.5*np.log(1+xi**2)),
        color="#2ca02c"))
    C.append(FrictionFunc("soft-clip",
        g=lambda xi: xi/np.sqrt(1+xi**2),
        g_prime=lambda xi: 1.0/(1+xi**2)**1.5,
        V=lambda xi: np.sqrt(1+xi**2), color="#d62728"))
    for a,b,col in [(4,1,"#9467bd"),(6,1,"#8c564b"),(2,0.1,"#e377c2")]:
        C.append(FrictionFunc(f"logosc({a},{b})",
            g=lambda xi,a=a,b=b: a*xi/(1+b*xi**2),
            g_prime=lambda xi,a=a,b=b: a*(1-b*xi**2)/(1+b*xi**2)**2,
            V=lambda xi,a=a,b=b: (a/(2*b))*np.log(1+b*xi**2), color=col))
    C.append(FrictionFunc("cubic-sat",
        g=lambda xi: 3*xi/(1+xi**2)**1.5,
        g_prime=lambda xi: 3*(1-2*xi**2)/(1+xi**2)**2.5,
        V=lambda xi: -3.0/np.sqrt(1+xi**2)+3.0, color="#7f7f7f"))
    return C

# ===========================================================================
# Part 1: Analytical
# ===========================================================================
def compute_gprime(f, Q, kT=1.0):
    if f.gprime_exact: return f.gprime_exact(Q)
    xi_max = min(200, max(30, 20/max(Q,0.01)))
    def inum(xi):
        v = -Q*f.V(xi)/kT
        return f.g_prime(xi)*np.exp(v) if v > -500 else 0.0
    def iden(xi):
        v = -Q*f.V(xi)/kT
        return np.exp(v) if v > -500 else 0.0
    n,_ = integrate.quad(inum, -xi_max, xi_max, limit=200)
    d,_ = integrate.quad(iden, -xi_max, xi_max, limit=200)
    return n/d if d > 1e-300 else 0.0

def omega_xi(f, Q, kT=1.0):
    gp = compute_gprime(f, Q, kT)
    return np.sqrt(gp/Q) if gp > 0 and Q > 0 else 0.0

def find_omega_max(f, kT=1.0):
    Qs = np.logspace(-2, 2, 200)
    omegas = np.array([omega_xi(f, Q, kT) for Q in Qs])
    idx = np.argmax(omegas)
    if idx <= 3:  # monotone decreasing
        tiny = [0.001, 0.005, 0.01]
        tiny_om = [omega_xi(f, Q, kT) for Q in tiny]
        best_i = np.argmax(tiny_om)
        if tiny_om[best_i] > omegas[idx]:
            return tiny_om[best_i], tiny[best_i]
    elif idx > 3 and idx < len(Qs)-3:
        try:
            res = optimize.minimize_scalar(lambda lq: -omega_xi(f, np.exp(lq), kT),
                bounds=(np.log(Qs[max(0,idx-10)]), np.log(Qs[min(len(Qs)-1,idx+10)])),
                method='bounded')
            return -res.fun, np.exp(res.x)
        except: pass
    return omegas[idx], Qs[idx]

def part1():
    print("="*60+"\nPART 1: Analytical frequency ceilings\n"+"="*60)
    cands = make_all()
    Q_arr = np.logspace(-2, 2, 200)
    results = {}
    for f in cands:
        t0 = time.time()
        om_curve = np.array([omega_xi(f, Q) for Q in Q_arr])
        gp_curve = np.array([compute_gprime(f, Q) for Q in Q_arr])
        om_max, Qstar = find_omega_max(f)
        # classify
        om_small = omega_xi(f, 0.01); om_mid = omega_xi(f, 1.0)
        finite_peak = om_small < om_mid * 0.95
        results[f.name] = dict(omega_max=float(om_max), Q_star=float(Qstar),
            omega_curve=om_curve.tolist(), gprime_curve=gp_curve.tolist(),
            has_finite_peak=finite_peak, elapsed_s=time.time()-t0)
        tag = "finite" if finite_peak else "sup"
        print(f"  {f.name:20s}  omega_max={om_max:.4f}  Q*={Qstar:.4f}  [{tag}]")
    results["Q_array"] = Q_arr.tolist()
    ranked = sorted([(k,v["omega_max"]) for k,v in results.items() if k!="Q_array"],
                    key=lambda x:-x[1])
    print("\nRanking:")
    for i,(n,o) in enumerate(ranked): print(f"  {i+1}. {n:20s}  {o:.4f}")
    return results, ranked

# ===========================================================================
# Part 2: Numerical (reduced scope)
# ===========================================================================
class HO:
    name="ho"; dim=1
    def __init__(self,w): self.w=w; self.kappas=np.array([w**2])
    def energy(self,q): return 0.5*self.w**2*q[0]**2
    def gradient(self,q): return np.array([self.w**2*q[0]])

def sim1(g_func, pot, Q, dt, nsteps, kT=1.0, seed=0, rec=1):
    rng=np.random.default_rng(seed); dim=pot.dim
    q=rng.normal(0,np.sqrt(kT),size=dim); p=rng.normal(0,np.sqrt(kT),size=dim)
    xi=0.0; h=0.5*dt; gU=pot.gradient(q)
    nr=nsteps//rec; qs=np.empty((nr,dim)); ri=0
    for s in range(nsteps):
        K=float(np.sum(p*p)); xi+=h*(K-dim*kT)/Q
        gv=g_func(xi); p*=np.clip(np.exp(-gv*h),1e-10,1e10); p-=h*gU
        q=q+dt*p; gU=pot.gradient(q)
        p-=h*gU; gv=g_func(xi); p*=np.clip(np.exp(-gv*h),1e-10,1e10)
        K=float(np.sum(p*p)); xi+=h*(K-dim*kT)/Q
        if (s+1)%rec==0 and ri<nr: qs[ri]=q; ri+=1
        if not np.isfinite(p).all(): qs[ri:]=np.nan; break
    return qs[:ri]

def acf_tau(x, c=5.0):
    x=np.asarray(x,float)-np.mean(x); n=len(x)
    if n<16 or np.std(x)<1e-12: return float(n)
    f=np.fft.fft(x,n=2*n); a=np.fft.ifft(f*np.conj(f))[:n].real; a/=a[0]
    tau=1.0
    for k in range(1,n//4):
        tau+=2*a[k]
        if k>=c*tau: break
    return max(tau,1.0)

def part2():
    print("\n"+"="*60+"\nPART 2: 1D HO validation (4 key candidates)\n"+"="*60)
    allf = {f.name:f for f in make_all()}
    keys = ["log-osc","tanh","soft-clip","logosc(6,1)"]
    omegas = [0.3, 1.0, 3.0, 10.0]
    Q_arr = np.logspace(-1.5, 1.5, 12)  # reduced
    nseeds=3; nsteps=50_000  # reduced
    results = {}
    for k in keys:
        f = allf[k]; results[k] = {}
        for w in omegas:
            pot=HO(w); dt=min(0.01, 0.3/w)
            best_tau=np.inf; best_Q=None; tbyQ=[]
            for Q in Q_arr:
                ts=[]
                for s in range(nseeds):
                    tr=sim1(f.g,pot,Q,dt,nsteps,seed=1000*s+7,rec=max(1,nsteps//10000))
                    v=tr[~np.isnan(tr[:,0])]
                    ts.append(acf_tau(v[:,0]**2) if len(v)>100 else 1e6)
                mt=float(np.mean(ts)); tbyQ.append(dict(Q=float(Q),tau=mt))
                if mt<best_tau: best_tau=mt; best_Q=float(Q)
            results[k][f"omega={w}"] = dict(best_Q=best_Q, best_tau=best_tau, tau_by_Q=tbyQ)
            print(f"  {k:20s}  w={w:5.1f}  Q*={best_Q:.4f}  tau={best_tau:.1f}")
    return results

# ===========================================================================
# Part 3: Head-to-head (top 2 + log-osc vs NHC)
# ===========================================================================
class AG:
    name="ag"
    def __init__(self,k): self.kappas=np.asarray(k,float); self.dim=len(k)
    def energy(self,q): return 0.5*float(np.sum(self.kappas*q*q))
    def gradient(self,q): return self.kappas*q

def sim_multi(g_func, pot, Qs, dt, nsteps, kT=1.0, seed=0, rec=1):
    rng=np.random.default_rng(seed); dim=pot.dim; Qs=np.asarray(Qs,float); N=len(Qs)
    q=rng.normal(0,1.0,size=dim)
    if hasattr(pot,'kappas'): q/=np.sqrt(np.maximum(pot.kappas,1e-6))
    p=rng.normal(0,np.sqrt(kT),size=dim); xi=np.zeros(N); h=0.5*dt
    gU=pot.gradient(q); nr=nsteps//rec; qs=np.empty((nr,dim)); ri=0
    for s in range(nsteps):
        K=float(np.sum(p*p)); xi+=h*(K-dim*kT)/Qs
        gt=sum(g_func(x) for x in xi); p*=np.clip(np.exp(-gt*h),1e-10,1e10); p-=h*gU
        q=q+dt*p; gU=pot.gradient(q)
        p-=h*gU; gt=sum(g_func(x) for x in xi); p*=np.clip(np.exp(-gt*h),1e-10,1e10)
        K=float(np.sum(p*p)); xi+=h*(K-dim*kT)/Qs
        if (s+1)%rec==0 and ri<nr: qs[ri]=q; ri+=1
        if not np.isfinite(p).all(): qs[ri:]=np.nan; break
    return qs[:ri]

def sim_nhc(pot, Qs, dt, nsteps, kT=1.0, seed=0, rec=1):
    rng=np.random.default_rng(seed); dim=pot.dim; Qs=np.asarray(Qs,float); M=len(Qs)
    q=rng.normal(0,1.0,size=dim)
    if hasattr(pot,'kappas'): q/=np.sqrt(np.maximum(pot.kappas,1e-6))
    p=rng.normal(0,np.sqrt(kT),size=dim); xi=np.zeros(M); h=0.5*dt
    gU=pot.gradient(q); nr=nsteps//rec; qs=np.empty((nr,dim)); ri=0
    def dxi(pv,xv):
        d=np.zeros(M); K=float(np.sum(pv*pv))
        d[0]=(K-dim*kT)/Qs[0]
        if M>1: d[0]-=xv[1]*xv[0]
        for i in range(1,M):
            d[i]=(Qs[i-1]*xv[i-1]**2-kT)/Qs[i]
            if i<M-1: d[i]-=xv[i+1]*xv[i]
        return d
    for s in range(nsteps):
        xi+=h*dxi(p,xi); p*=np.clip(np.exp(-xi[0]*h),1e-10,1e10); p-=h*gU
        q=q+dt*p; gU=pot.gradient(q)
        p-=h*gU; p*=np.clip(np.exp(-xi[0]*h),1e-10,1e10); xi+=h*dxi(p,xi)
        if (s+1)%rec==0 and ri<nr: qs[ri]=q; ri+=1
        if not np.isfinite(p).all(): qs[ri:]=np.nan; break
    return qs[:ri]

def tau_q2(tr):
    v=tr[~np.isnan(tr[:,0])]
    if len(v)<64: return 1e6
    return float(np.mean([acf_tau(v[:,d]**2) for d in range(v.shape[1])]))

def mode_cross(tr,gmm):
    v=tr[~np.isnan(tr[:,0])]
    if len(v)==0: return 0
    d2=np.sum((v[:,None,:]-gmm.centers[None,:,:])**2,axis=2)
    a=np.argmin(d2,axis=1)
    return int(np.sum(a[1:]!=a[:-1]))

def part3(top_names):
    print("\n"+"="*60+"\nPART 3: Benchmarks\n"+"="*60)
    allf={f.name:f for f in make_all()}; kT=1.0; ns=5; nfe=200_000; kr=100.0
    kappas=np.array([kr**(i/4.0) for i in range(5)])
    pot=AG(kappas); dt=0.05/np.sqrt(kr); Nth=5
    r5d={}
    # NHC
    taus=[]
    for s in range(ns):
        tr=sim_nhc(pot,np.ones(Nth),dt,nfe,kT=kT,seed=5000+s,rec=4)
        taus.append(tau_q2(tr))
    r5d["NHC(M=5)"]=dict(mean_tau=float(np.mean(taus)),std_tau=float(np.std(taus)))
    print(f"  NHC(M=5): tau={np.mean(taus):.1f}+/-{np.std(taus):.1f}")
    for nm in top_names:
        f=allf[nm]; Qs=np.exp(np.linspace(np.log(0.01),np.log(10),Nth))
        taus=[]
        for s in range(ns):
            tr=sim_multi(f.g,pot,Qs,dt,nfe,kT=kT,seed=5000+s,rec=4)
            taus.append(tau_q2(tr))
        r5d[nm]=dict(mean_tau=float(np.mean(taus)),std_tau=float(np.std(taus)),Qs=Qs.tolist())
        print(f"  {nm:20s}: tau={np.mean(taus):.1f}+/-{np.std(taus):.1f}")
    # GMM
    gmm=GaussianMixture2D(n_modes=5,radius=3.0,sigma=0.5); dtg=0.02; nsg=200_000; Ng=3
    rgmm={}
    cr=[]
    for s in range(ns):
        tr=sim_nhc(gmm,np.ones(Ng),dtg,nsg,kT=kT,seed=6000+s,rec=4)
        cr.append(mode_cross(tr,gmm))
    rgmm["NHC(M=3)"]=dict(mean_cr=float(np.mean(cr)),std_cr=float(np.std(cr)))
    print(f"\n  GMM NHC(M=3): cr={np.mean(cr):.1f}+/-{np.std(cr):.1f}")
    for nm in top_names:
        f=allf[nm]; Qs=np.exp(np.linspace(np.log(0.1),np.log(10),Ng))
        cr=[]
        for s in range(ns):
            tr=sim_multi(f.g,gmm,Qs,dtg,nsg,kT=kT,seed=6000+s,rec=4)
            cr.append(mode_cross(tr,gmm))
        rgmm[nm]=dict(mean_cr=float(np.mean(cr)),std_cr=float(np.std(cr)),Qs=Qs.tolist())
        print(f"  {nm:20s}: cr={np.mean(cr):.1f}+/-{np.std(cr):.1f}")
    return dict(gaussian_5d=r5d, gmm_2d=rgmm)

# ===========================================================================
# Figures
# ===========================================================================
def make_figs(a_res, n_res, b_res, ranked):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    cands=make_all(); Q_arr=np.array(a_res["Q_array"])
    cm={f.name:f.color for f in cands}

    # Fig 1: frequency ceilings
    fig=plt.figure(figsize=(16,5))
    gs=fig.add_gridspec(1,3,width_ratios=[2,1.2,1.2],wspace=0.35)

    ax=fig.add_subplot(gs[0])
    for f in cands:
        r=a_res[f.name]; om=np.array(r["omega_curve"])
        ax.plot(Q_arr,om,label=f.label,color=f.color,lw=1.8)
        if r["has_finite_peak"]:
            ax.plot(r["Q_star"],r["omega_max"],"o",color=f.color,ms=5,zorder=5)
    ax.set_xscale("log"); ax.set_xlabel("$Q$",fontsize=14)
    ax.set_ylabel("$\\omega_\\xi(Q)$",fontsize=13)
    ax.set_title("(a) Thermostat frequency vs $Q$",fontsize=14)
    ax.legend(fontsize=7.5,loc="upper right",framealpha=0.9,ncol=2)
    ax.set_xlim(1e-2,1e2); ax.set_ylim(0,2.5)
    ax.tick_params(labelsize=12); ax.grid(True,alpha=0.3)
    ax.axhline(0.732,color="#1f77b4",ls=":",alpha=0.5,lw=1)
    ax.axhline(1.0,color="#ff7f0e",ls=":",alpha=0.5,lw=1)
    ax.text(3e1,0.75,"log-osc",fontsize=8,color="#1f77b4",alpha=0.7)
    ax.text(3e1,1.02,"tanh",fontsize=8,color="#ff7f0e",alpha=0.7)

    ax2=fig.add_subplot(gs[1])
    names=[r[0] for r in ranked]; vals=[r[1] for r in ranked]
    cols=[cm.get(n,"#333") for n in names]
    ax2.barh(range(len(names)),vals,color=cols,edgecolor="white",lw=0.5)
    ax2.set_yticks(range(len(names))); ax2.set_yticklabels(names,fontsize=9)
    ax2.set_xlabel("$\\omega_{max}$",fontsize=14)
    ax2.set_title("(b) Ceiling ranking",fontsize=14)
    ax2.invert_yaxis(); ax2.tick_params(labelsize=11)
    for i,v in enumerate(vals): ax2.text(v+0.02,i,f"{v:.3f}",va="center",fontsize=8)
    ax2.set_xlim(0,max(vals)*1.15)
    ax2.axvline(0.732,color="#1f77b4",ls=":",alpha=0.5)

    ax3=fig.add_subplot(gs[2])
    for f in cands:
        gp=np.array(a_res[f.name]["gprime_curve"])
        ax3.plot(Q_arr,gp,color=f.color,lw=1.5,label=f.label)
    ax3.set_xscale("log"); ax3.set_xlabel("$Q$",fontsize=14)
    ax3.set_ylabel("$\\langle g\\prime \\rangle_Q$",fontsize=13)
    ax3.set_title("(c) Coupling strength",fontsize=14)
    ax3.set_xlim(1e-2,1e2); ax3.tick_params(labelsize=11); ax3.grid(True,alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR,"fig1_frequency_ceilings.png"),dpi=200,bbox_inches="tight")
    plt.close(fig); print("Saved fig1")

    # Fig 2: tau vs Q on 1D HO
    if n_res:
        fig2,axes=plt.subplots(1,4,figsize=(16,4.5),sharey=True)
        ws=[0.3,1.0,3.0,10.0]
        for j,w in enumerate(ws):
            ax=axes[j]; key=f"omega={w}"
            for nm in n_res:
                if key in n_res[nm]:
                    d=n_res[nm][key]
                    ax.plot([x["Q"] for x in d["tau_by_Q"]],
                            [x["tau"] for x in d["tau_by_Q"]],
                            "o-",color=cm.get(nm,"#333"),label=nm,ms=3,lw=1.2)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel("$Q$",fontsize=12); ax.set_title(f"$\\omega={w}$",fontsize=13)
            ax.grid(True,alpha=0.3); ax.tick_params(labelsize=10)
            if j==0: ax.set_ylabel("$\\tau_{int}$",fontsize=12); ax.legend(fontsize=7)
        plt.suptitle("$\\tau_{int}(q^2)$ vs $Q$ on 1D harmonic oscillator",fontsize=14,y=1.02)
        plt.tight_layout()
        fig2.savefig(os.path.join(FIG_DIR,"fig2_q_opt_comparison.png"),dpi=200,bbox_inches="tight")
        plt.close(fig2); print("Saved fig2")

    # Fig 3: benchmark bars
    if b_res:
        fig3,(a1,a2)=plt.subplots(1,2,figsize=(14,5))
        d5=b_res["gaussian_5d"]; n5=list(d5.keys())
        a1.bar(range(len(n5)),[d5[n]["mean_tau"] for n in n5],
               yerr=[d5[n]["std_tau"] for n in n5],
               color=[cm.get(n,"#999") for n in n5],edgecolor="white",capsize=3)
        a1.set_xticks(range(len(n5))); a1.set_xticklabels(n5,fontsize=9,rotation=30,ha="right")
        a1.set_ylabel("$\\tau_{int}$",fontsize=13); a1.set_title("(a) 5D Anisotropic",fontsize=13)

        dg=b_res["gmm_2d"]; ng=list(dg.keys())
        a2.bar(range(len(ng)),[dg[n]["mean_cr"] for n in ng],
               yerr=[dg[n]["std_cr"] for n in ng],
               color=[cm.get(n,"#999") for n in ng],edgecolor="white",capsize=3)
        a2.set_xticks(range(len(ng))); a2.set_xticklabels(ng,fontsize=9,rotation=30,ha="right")
        a2.set_ylabel("Mode crossings",fontsize=13); a2.set_title("(b) 2D GMM",fontsize=13)

        plt.tight_layout()
        fig3.savefig(os.path.join(FIG_DIR,"fig3_benchmark.png"),dpi=200,bbox_inches="tight")
        plt.close(fig3); print("Saved fig3")

# ===========================================================================
def main():
    t0=time.time()
    a_res, ranked = part1()
    n_res = part2()
    # Top 2 finite-peak + tanh + log-osc
    top = ["tanh","soft-clip","log-osc"]
    print(f"\nBenchmark candidates: {top}")
    b_res = part3(top)
    total = time.time()-t0

    out = dict(analytical=a_res, numerical_1d=n_res, benchmark=b_res,
               ranking=[dict(name=r[0],omega_max=r[1]) for r in ranked],
               top_candidates=top, best_omega_max=ranked[0][1],
               best_name=ranked[0][0], total_time_s=total)
    with open(os.path.join(OUT_DIR,"results.json"),"w") as f:
        json.dump(out,f,indent=2,default=float)
    print(f"\nSaved results.json")
    make_figs(a_res, n_res, b_res, ranked)
    print(f"\nTotal: {total:.1f}s")
    print(f"\nHEADLINE: Best={ranked[0][0]} omega_max={ranked[0][1]:.4f}")
    print(f"          Log-osc omega_max={a_res['log-osc']['omega_max']:.4f}")
    return out

if __name__=="__main__": main()
