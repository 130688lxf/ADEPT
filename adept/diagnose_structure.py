#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Population-structure diagnosis, run before step 1
================================================================
Changes nothing in step 1; it answers three questions.

  1. Is the reference population stratified at all, and into how many strata?
  2. How strong is that stratification (the leading eigenvalue, and is there a
     spectral gap)?
  3. The decisive one: is the stratification correlated with the phenotypes,
     the shell-colour traits in particular? That determines whether PC or Q
     correction is safe, or whether it would delete real heritability.

Decision logic, printed as the conclusion at the end of the run:
  - no stratification                      -> step 1 uses an intercept only
    (0 PCs); the random K_A term handles all relatedness
  - stratified but uncorrelated with the phenotypes -> correction is safe; use
    a small number of Q columns or PCs
  - stratified AND strongly correlated with a trait -> dangerous. The
    between-line differences in that trait are quite possibly REAL genetic
    variance; correcting would delete it as if it were structure and push h2
    towards zero. Use the known line labels and the rearing design to decide
    whether this is a genetic difference or environmental confounding, run a
    sensitivity analysis, and do not add PCs blindly.

Dependencies: numpy, scipy, pandas; matplotlib optionally for the figure;
plink and admixture optionally for the ADMIXTURE cross-validation, which is
skipped automatically when they are absent and replaced by the eigenvalue
diagnosis.
================================================================
"""
import os, shutil, subprocess, re, warnings
import numpy as np
import pandas as pd
from scipy.linalg import eigh
warnings.filterwarnings("ignore")

# ==================== Configuration ====================
REF_VCF    = "ref_all_phased.vcf"
PHENO_FILE = "trait_imputed.csv"
TRAITS = ["phe1","phe2","phe3","phe4","phe5","phe6","phe7","phe8","phe14","phe15"]
COLOR_TRAITS = ["phe14","phe15"]          # shell colour is the most likely to track line
OUT_DIR    = "structure_diagnosis"
os.makedirs(OUT_DIR, exist_ok=True)

N_TOP_PC     = 10          # how many leading PCs are checked against the phenotypes
ADMIX_K_RANGE = range(1, 7) # candidate ancestral population counts; K=1 means none
ADMIX_BIN    = "admixture"
PLINK_BIN    = "plink"
MAKE_PLOTS   = True

# ==================== I/O, identical to step 1 ====================
def normalize_id(s):
    s = str(s).strip().lstrip("\ufeff").replace("\r","").replace("\n","")
    if s.endswith(".0") and s[:-2].replace("-","").replace("_","").isalnum():
        s = s[:-2]
    return s

def read_phased_vcf(path):
    ids=None; chrom=[]; pos=[]; snp=[]; h1=[]; h2=[]
    with open(path) as f:
        for line in f:
            line=line.rstrip("\r\n")
            if line.startswith("##"): continue
            p=line.split("\t")
            if line.startswith("#CHROM"):
                ids=np.array([normalize_id(x) for x in p[9:]],dtype=str); continue
            chrom.append(p[0]); pos.append(int(p[1])); snp.append(p[2])
            a=[]; b=[]
            for g in p[9:]:
                gt=g.split(":")[0]
                x,y=(gt.split("|") if "|" in gt else gt.split("/"))
                a.append(0 if x=="0" else 1); b.append(0 if y=="0" else 1)
            h1.append(a); h2.append(b)
    H1=np.array(h1,dtype=np.int8).T; H2=np.array(h2,dtype=np.int8).T
    X=(H1+H2).astype(float)
    return ids,np.array(chrom),np.array(pos),np.array(snp),H1,H2,X

def grm_a(X):
    p=X.mean(0)/2.0
    Z=X-2.0*p
    c=2.0*np.sum(p*(1.0-p))+1e-12
    K=(Z@Z.T)/c
    return K/np.mean(np.diag(K))

def pca_from_grm(K,n):
    ev,evec=eigh(K); o=np.argsort(ev)[::-1][:n]
    return evec[:,o]*np.sqrt(np.maximum(ev[o],0))

# ==================== Identifier matching, as in step 1 ====================
def match_ids(ids, phe):
    vcf_ids=ids.copy()
    inter=set(vcf_ids)&set(phe.index)
    if len(inter)<0.5*len(vcf_ids):
        alt=np.array([i.split("_")[-1] for i in vcf_ids])
        if len(set(alt)&set(phe.index))>len(inter):
            vcf_ids=alt; inter=set(vcf_ids)&set(phe.index)
    keep=[k for k,i in enumerate(vcf_ids) if i in phe.index]
    return np.array(keep), vcf_ids

# ==================== ADMIXTURE cross-validation (optional) ====================
def run_admixture_cv(vcf_path, k_range, workdir):
    if shutil.which(PLINK_BIN) is None or shutil.which(ADMIX_BIN) is None:
        print("  [skipped] plink or admixture not found;")
        print("         the eigenvalue and PC-phenotype diagnosis is used instead. The")
        print("         conclusion is equally sound, with one fewer cross-check.\n"
              "         To enable it: conda install -c bioconda plink admixture")
        return None
    os.makedirs(workdir, exist_ok=True)
    bed=os.path.join(workdir,"ref")
    try:
        subprocess.run([PLINK_BIN,"--vcf",vcf_path,"--make-bed","--double-id",
                        "--allow-extra-chr","--out",bed],
                       check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        # LD pruning; ADMIXTURE assumes near-independent markers
        subprocess.run([PLINK_BIN,"--bfile",bed,"--indep-pairwise","50","5","0.2",
                        "--allow-extra-chr","--out",bed+"_pr"],
                       check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        subprocess.run([PLINK_BIN,"--bfile",bed,"--extract",bed+"_pr.prune.in",
                        "--make-bed","--allow-extra-chr","--out",bed+"_p"],
                       check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        # recode chromosomes as integers, which is all ADMIXTURE accepts
        bim=bed+"_p.bim"; lines=open(bim).read().splitlines(); cmap={}; out=[]
        for ln in lines:
            fld=ln.split("\t")
            if not fld[0].lstrip("-").isdigit():
                if fld[0] not in cmap: cmap[fld[0]]=len(cmap)+1
                fld[0]=str(cmap[fld[0]])
            out.append("\t".join(fld))
        open(bim,"w").write("\n".join(out)+"\n")
        cv={}
        for k in k_range:
            logf=os.path.join(workdir,f"k{k}.log")
            with open(logf,"w") as fo:
                subprocess.run([ADMIX_BIN,"--cv","-s","1","ref_p.bed",str(k)],
                               cwd=workdir,check=True,stdout=fo,stderr=subprocess.STDOUT)
            for line in open(logf):
                m=re.search(r"CV error \(K=(\d+)\):\s*([\d.]+)",line)
                if m: cv[int(m.group(1))]=float(m.group(2))
        return cv
    except subprocess.CalledProcessError as e:
        print(f"  [skipped] ADMIXTURE/plink failed ({e}); using the eigenvalue diagnosis.")
        return None

# ==================== Main ====================
def main():
    print("="*70); print("Population-structure diagnosis"); print("="*70)
    ids,chrom,pos,snp,H1,H2,X = read_phased_vcf(REF_VCF)
    phe=pd.read_csv(PHENO_FILE,dtype=str)
    phe["IID"]=phe["IID"].map(normalize_id)
    phe=phe.drop_duplicates("IID").set_index("IID")
    for t in TRAITS: phe[t]=pd.to_numeric(phe[t],errors="coerce")

    keep,vcf_ids = match_ids(ids,phe)
    ids2=vcf_ids[keep]; X=X[keep]
    Y=phe.reindex(ids2)[TRAITS].values.astype(float)
    ok=~np.isnan(Y).any(1)
    ids2=ids2[ok]; X=X[ok]; Y=Y[ok]
    n,m=X.shape
    print(f"  usable individuals n={n}, SNP={m}\n")

    K=grm_a(X)
    ev=np.sort(eigh(K,eigvals_only=True))[::-1]   # descending eigenvalues, mean about 1
    PC=pca_from_grm(K,N_TOP_PC)

    # ---------- diagnosis 1: the eigenvalue spectrum ----------
    print("-"*70); print("[diagnosis 1] K_A eigenvalue spectrum. In a homogeneous population every\n"
          "              eigenvalue is near 1; under stratification the leading ones stand out")
    print("-"*70)
    print("  first 10 eigenvalues:", np.round(ev[:10],3))
    lam1=ev[0]; gap=ev[0]/ev[1] if len(ev)>1 else np.nan
    # broken-stick, reported to show why it selects far too many components
    bs=np.array([1.0/n*np.sum(1.0/np.arange(i+1,n+1)) for i in range(20)])
    pct=ev[:20]/ev.sum()
    bs_count=int(np.sum(pct>bs))
    # the largest drop among the first ten eigenvalues; the gap is far more
    # reliable than the absolute magnitude
    top=ev[:min(10,len(ev))]
    ratios=top[:-1]/top[1:]
    gap_at=int(np.argmax(ratios))+1     # the largest gap falls after this eigenvalue
    max_ratio=float(ratios.max())
    print(f"  PC1 eigenvalue = {lam1:.3f}  (the absolute value depends on the number of\n"
          f"    SNPs and the scaling, so treat it as indicative only)")
    print(f"  largest spectral gap after eigenvalue {gap_at}, drop ratio {max_ratio:.2f}")
    print(f"    (a homogeneous population has no such drop; under stratification the\n"
          f"     first K-1 eigenvalues separate clearly)")
    print(f"  broken-stick would call {bs_count} PCs significant. That count includes")
    print(f"    family genetic signal and must not be used as the number of PCs to remove.")

    # ---------- diagnosis 2: ADMIXTURE CV (optional) ----------
    print("\n"+"-"*70); print("[diagnosis 2] ADMIXTURE cross-validation for K"); print("-"*70)
    cv=run_admixture_cv(REF_VCF, ADMIX_K_RANGE, os.path.join(OUT_DIR,"admix"))
    admix_k=None
    if cv:
        print("  CV error:", {k:round(v,4) for k,v in sorted(cv.items())})
        admix_k=min(cv,key=cv.get)
        print(f"  -> CV selects K = {admix_k}  "
              f"({'no discrete strata' if admix_k==1 else f'{admix_k} ancestral populations'})")

    # ---------- diagnosis 3: PCs against the phenotypes, the decisive one ----------
    print("\n"+"-"*70)
    print("[diagnosis 3] how much trait variance the leading PCs explain (R2), which\n"
          "              decides whether correcting is safe")
    print("-"*70)
    Yc=(Y-Y.mean(0))/Y.std(0)
    r2=np.zeros((N_TOP_PC,len(TRAITS)))
    for pi in range(N_TOP_PC):
        x=PC[:,pi]; x=(x-x.mean())/x.std()
        for ti in range(len(TRAITS)):
            r2[pi,ti]=np.corrcoef(x,Yc[:,ti])[0,1]**2
    r2df=pd.DataFrame(r2, index=[f"PC{i+1}" for i in range(N_TOP_PC)], columns=TRAITS)
    r2df.to_csv(os.path.join(OUT_DIR,"pc_trait_r2.csv"))
    # cumulative variance explained per trait by the first k PCs
    cum_r2={}
    for ti,t in enumerate(TRAITS):
        for kk in [2,5,10]:
            Xk=np.column_stack([np.ones(n),PC[:,:kk]])
            beta,_,_,_=np.linalg.lstsq(Xk,Yc[:,ti],rcond=None)
            resid=Yc[:,ti]-Xk@beta
            cum_r2[(t,kk)]=1-resid.var()/Yc[:,ti].var()
    print("  R2 of each trait on an intercept plus the first k PCs:")
    print(f"    {'trait':<8}{'2 PCs':>8}{'5 PCs':>8}{'10 PCs':>8}")
    color_risk=[]
    for t in TRAITS:
        flag=""
        if t in COLOR_TRAITS and cum_r2[(t,5)]>0.10: flag="  <- colour largely explained by PCs: high risk"
        if cum_r2[(t,5)]>0.15: color_risk.append(t)
        print(f"    {t:<8}{cum_r2[(t,2)]:>8.3f}{cum_r2[(t,5)]:>8.3f}{cum_r2[(t,10)]:>8.3f}{flag}")

    # ---------- figure ----------
    if MAKE_PLOTS:
        try:
            import matplotlib; matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig,ax=plt.subplots(1,3,figsize=(15,4))
            ax[0].plot(range(1,21),ev[:20],"o-"); ax[0].axhline(1,ls="--",c="gray")
            ax[0].set_title("Scree (top20 eigenvalues)"); ax[0].set_xlabel("PC")
            c = Yc[:,TRAITS.index(COLOR_TRAITS[0])] if COLOR_TRAITS[0] in TRAITS else None
            sc=ax[1].scatter(PC[:,0],PC[:,1],c=c,cmap="coolwarm",s=12)
            ax[1].set_xlabel("PC1"); ax[1].set_ylabel("PC2")
            ax[1].set_title(f"PC1-PC2 (color = {COLOR_TRAITS[0]})"); plt.colorbar(sc,ax=ax[1])
            im=ax[2].imshow(r2,aspect="auto",cmap="Reds",vmin=0,vmax=max(0.2,r2.max()))
            ax[2].set_xticks(range(len(TRAITS))); ax[2].set_xticklabels(TRAITS,rotation=90)
            ax[2].set_yticks(range(N_TOP_PC)); ax[2].set_yticklabels([f"PC{i+1}" for i in range(N_TOP_PC)])
            ax[2].set_title("R²(PC, trait)"); plt.colorbar(im,ax=ax[2])
            plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR,"structure_diagnosis.png"),dpi=120)
            print(f"\n  figure written to {OUT_DIR}/structure_diagnosis.png")
        except Exception as e:
            print(f"  (figure skipped: {e})")

    # ---------- conclusion ----------
    print("\n"+"="*70); print("Conclusion and recommendation"); print("="*70)
    stratified = (max_ratio>3.0) or (admix_k is not None and admix_k>=2)
    if not stratified:
        print("  No clear stratification (no drop in the eigenvalue spectrum"
              + (", ADMIXTURE selects K=1" if admix_k==1 else "")+").")
        print("    -> step 1 should use an intercept only, 0 PCs. The random K_A term")
        print("       already absorbs family relatedness; PC fixed effects are neither\n"
              "       needed nor appropriate.")
    else:
        print(f"  Stratification detected (a {max_ratio:.1f}x drop after eigenvalue {gap_at}"
              + (f", ADMIXTURE K={admix_k}" if admix_k else "")+"）。")
        if color_risk:
            print(f"    However, the leading PCs explain much of these traits: {color_risk}")
            print("       Those are quite possibly REAL genetic differences between lines")
            print("       (the gold and purple lines genuinely differ in shell colour).")
            print("       Correcting on PCs or Q would delete that genetic variance as if it")
            print("       were structure and drive h2 towards zero. Instead: (1) obtain the")
            print("       line labels and determine whether the lines were reared together.")
            print("       Common rearing means the difference is genetic and no correction")
            print("       should be applied; separate rearing means possible environmental\n"
                  "       confounding and the line should enter as a fixed factor.\n"
                  "       (2) In either case run a sensitivity analysis reporting h2 and the\n"
                  "       key genetic correlations under 0 PCs, K-1 Q columns and line as a\n"
                  "       fixed factor.")
        else:
            print("    The stratification is not strongly correlated with any trait, so")
            print(f"    correcting is safe and cheap. Step 1 can use the K-1="
                  f"{('%d'%(admix_k-1)) if admix_k else '(K-1)'} ADMIXTURE Q columns as")
            print("    fixed effects, or equivalently two or three PCs. A sensitivity\n"
                  "    analysis is still recommended.")
    print("\n  see", OUT_DIR+"/pc_trait_r2.csv for the PC-by-trait matrix")
    print("="*70)

if __name__=="__main__":
    main()
