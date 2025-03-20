(* ::Package:: *)

(* ::Text:: *)
(*The equation indexing refers to https://arxiv.org/pdf/1708.06177*)
(*For the .m file, replace NotebookDirectory[] with DirectoryName[$InputFileName]*)
(*Type echo "wolframscript -file ToF.m" | at now to run it on Linux detached from the terminal*)
(**)


(* ::Section:: *)
(*Definitions*)


(* ::Input::Initialization:: *)
ToFOrder =10;                      (*Theory of Figures will be computed until order ToFOrder*)
DiffRotOrder = 22;          (*Rotational parameters \[Alpha] will be included until Subscript[\[Alpha], DiffRotOrder] for differential rotation*)
Subscript[\[Rho], av]    =M/((4\[Pi])/3 Rm^3);
l        =z*Rm;
(*Variable names used in numerical algorithm:*)
ss =List[];SS=List[];SSdash=List[];alphas=List[];
For[i=0,i<2*ToFOrder +1,i++, ss=Append[ss,ToExpression["s"<>ToString[i]]]];
For[i=0,i<2*ToFOrder +1,i++, SS=Append[SS,ToExpression["S"<>ToString[i]]]];
For[i=0,i<2*ToFOrder +1,i++, SSdash=Append[SSdash,ToExpression["S"<>ToString[i]<>"p"]]];
For[i=0,i<DiffRotOrder +1,i++, alphas=Append[alphas,ToExpression["alpha"<>ToString[i]]]];


(* ::Subsection:: *)
(*Core function definitions up to relevant order*)


(* ::Input::Initialization:: *)
PowerLawfn[n_,x_]                 =Normal[Series[(1+x)^(n+3),              {x,0,ToFOrder}]]/.x^\[Beta]_->f[x,\[Beta]];(*used in equation (B.10) for Subscript[f, n](z) *)
PowerLawfdashn[n_,x_]        =Normal[Series[(1+x)^(2-n),              {x,0,ToFOrder}]]/.x^\[Beta]_->f[x,\[Beta]];(*used in equation (B.10) for Subscript[f', n](z) *)
PowerLawfdash2[x_]                 =Normal[Series[Log[1+x],            {x,0,ToFOrder}]]/.x^\[Beta]_->f[x,\[Beta]];(*used in equation (B.10) for Subscript[f', 2](z) *)
PowerLawDn[n_,x_,\[Alpha]_]         =Normal[Series[(\[Alpha]+x)^(-n-1),            {x,0,ToFOrder}]]/.x^\[Beta]_->f[x,\[Beta]];  (*used in equation (B.5), in front of Subscript[D, 2n](r)*)
PowerLawDdashn[n_,x_,\[Alpha]_]=Normal[Series[(\[Alpha]+x)^n,                 {x,0,ToFOrder}]]/.x^\[Beta]_->f[x,\[Beta]];  (*used in equation (B.5), in front of Subscript[D', 2n](r)*)



(* ::Subsection:: *)
(*Insert r[l,\[Mu]] into core function definitions up to relevant order*)


(* ::Subsubsection:: *)
(*Preliminary definition of r[l,\[Mu]] with unconstrained Subscript[s, 0]*)


(* ::Input::Initialization:: *)
rpre[l_,\[Mu]_]=l(1+s[0,l]+Sum[OR^n*s[2n,l]*LegendreP[2n,\[Mu]],{n,1,ToFOrder}]); (*equation (B.1)*) (*OR is a helper parameter that we use to only calculate expressions up to the relevant order, Subscript[s, 0] will get replaced so no OR parameter necessary*)
Print["\!\(\*SubscriptBox[\(R\), \(equatorial\)]\)/\!\(\*SubscriptBox[\(R\), \(mean\)]\)=",FortranForm[rpre[l,0]/l/.OR->1/.s[\[Alpha]_,z Rm]->Indexed[ss,\[Alpha]+1]]]; 
Export[FileNameJoin[{DirectoryName[$InputFileName],"R_ratio.txt"}],FortranForm[rpre[l,0]/l/.OR->1/.s[\[Alpha]_,z Rm]->Indexed[ss,\[Alpha]+1]],"Text"];


(* ::Subsubsection:: *)
(*Condition of equal volume*)


(* ::Input::Initialization:: *)
Zero                     = Expand[(Integrate[rpre[z*Rm,\[Mu]]^3/(z*Rm)^3,{\[Mu],0,1}] -1)];(*This expression should be zero*)
solution            = Solve[Zero==0,s[0,Rm z]];
s0                          =Expand[Normal[Series[s[0,Rm z]/.solution[[1]],{OR,0,ToFOrder}]]];
s0fasts=List[];
(*Calculate powers of s0 in a fast way and storing them in a list:*)
Print["Expanding s0fasts..."]
For[i=1,i<Max[ToFOrder,DiffRotOrder] +1,i++, 
res =s0/.OR^\[Alpha]_/;\[Alpha]>ToFOrder+1-i->0;
time=AbsoluteTiming[s0fasts=Append[s0fasts,Expand[res^i]/.OR^\[Alpha]_/;\[Alpha]>ToFOrder->0]];
Print[N[i/Max[ToFOrder,DiffRotOrder]*100,3],"% complete. Step took ", time[[1]]," seconds."];];
Print["\!\(\*SubscriptBox[\(s\), \(0\)]\)=",s0/.s[\[Alpha]_,z Rm]->Subscript[s, \[Alpha]]/.OR->1];
Print["\!\(\*SubscriptBox[\(s\), \(0\)]\)=",FortranForm[FullSimplify[s0]/.OR->1/.s[\[Alpha]_,z Rm]->Indexed[ss,\[Alpha]+1]]];
Export[FileNameJoin[{DirectoryName[$InputFileName],"new_s0.txt"}],FortranForm[FullSimplify[s0]/.OR->1/.s[\[Alpha]_,z Rm]->Indexed[ss,\[Alpha]+1]],"Text"];


(* ::Subsubsection:: *)
(*Actual definition of r[l,\[Mu]] and \[CapitalSigma][l,\[Mu]]*)


(* ::Input::Initialization:: *)
r[l_,\[Mu]_]=l(1+s0+Sum[OR^n*s[2n,l]*LegendreP[2n,\[Mu]],{n,1,ToFOrder}])(*equation (B.1)*) (*OR is a helper parameter that we use to only calculate expressions up to the relevant order, Subscript[s, 0] will get replaced so no OR parameter necessary*)
\[CapitalSigma][l_,\[Mu]_]=s0+Sum[OR^n*s[2n,l]*LegendreP[2n,\[Mu]],{n,1,ToFOrder}];


(* ::Subsubsection:: *)
(*Fast calculation of r[l,\[Mu]] and \[CapitalSigma][l,\[Mu]]*)


(* ::Input::Initialization:: *)
(*If the following is confusing, check the slow calculation below*)
fast= x+q[0] + Sum[OR^n*q[2n],{n,1,ToFOrder}];
rfasts = List[];
\[CapitalSigma]fasts = List[];
Print["Expanding fast with placeholders x and q..."]
For[i=1,i<Max[ToFOrder,DiffRotOrder] +1,i++, 
time=AbsoluteTiming[res =Expand[(fast)^i]/.OR^\[Alpha]_/;\[Alpha]>ToFOrder->0; 
rfasts=Append[rfasts,(l^i*res)/.x->1];
\[CapitalSigma]fasts=Append[\[CapitalSigma]fasts,res/.x->0]];
Print[N[i/Max[ToFOrder,DiffRotOrder]*100,3],"% complete. Step took ",time[[1]]," seconds."];];
Print["Replacing q[0] with s0fasts..."]
For[i=1,i<Max[ToFOrder,DiffRotOrder] +1,i++, 
time=AbsoluteTiming[rfasts[[i]]=Expand[rfasts[[i]]/.q[0]^\[Alpha]_->Indexed[s0fasts,\[Alpha]]/.q[0]->s0fasts[[1]]]/.OR^\[Alpha]_/;\[Alpha]>ToFOrder->0;
\[CapitalSigma]fasts[[i]]=Expand[\[CapitalSigma]fasts[[i]]/.q[0]^\[Alpha]_->Indexed[s0fasts,\[Alpha]]/.q[0]->s0fasts[[1]]]/.OR^\[Alpha]_/;\[Alpha]>ToFOrder->0];Print[N[i/Max[ToFOrder,DiffRotOrder]*100,3],"% complete. Step took ",time[[1]]," seconds."];];
Print["Replacing q[n] with s[n,l]*LegendreP[n,\[Mu]]..."]
For[i=1,i<Max[ToFOrder,DiffRotOrder] +1,i++, 
time=AbsoluteTiming[rfasts[[i]]=Expand[rfasts[[i]]/.q[n_]^\[Alpha]_->s[n,l]^\[Alpha]*LegendreP[n,\[Mu]]^\[Alpha]/.q[n_]->s[n,l]*LegendreP[n,\[Mu]]];
\[CapitalSigma]fasts[[i]]=Expand[\[CapitalSigma]fasts[[i]]/.q[n_]^\[Alpha]_->s[n,l]^\[Alpha]*LegendreP[n,\[Mu]]^\[Alpha]/.q[n_]->s[n,l]*LegendreP[n,\[Mu]]]];
Print[N[i/Max[ToFOrder,DiffRotOrder]*100,3],"% complete. Step took ",time[[1]]," seconds."];];
\[CapitalSigma]s = \[CapitalSigma]fasts;
rs = rfasts;


(* ::Subsubsection:: *)
(*Slow calculation of r[l,\[Mu]] and \[CapitalSigma][l,\[Mu]] (optional)*)


(* ::Input:: *)
(*(*Slow calculation that is more straightforward, uncomment to check:*)*)
(*(*\[CapitalSigma]s = List[];*)
(*Print["Expanding \[CapitalSigma] directly..."]*)
(*For[i=1,i<Max[ToFOrder,DiffRotOrder] +1,i++, time=AbsoluteTiming[\[CapitalSigma]s=Append[\[CapitalSigma]s,Expand[(\[CapitalSigma][l,\[Mu]])^i]/.OR^\[Alpha]_/;\[Alpha]>ToFOrder->0]];Print[N[i/Max[ToFOrder,DiffRotOrder]*100,3],"% complete. Step took ",time[[1]]," seconds."];];*)
(*Print["Ensuring that \[CapitalSigma] and \[CapitalSigma]fast are consistent..."]*)
(*For[i=1,i<Max[ToFOrder,DiffRotOrder] +1,i++,Print["This should be zero: ", \[CapitalSigma]fasts[[i]]-\[CapitalSigma]s[[i]]]];*)
(*\[CapitalSigma]s = \[CapitalSigma]fasts;*)
(*rs=List[];*)
(*Print["Expanding r directly..."]*)
(*For[i=1,i<Max[ToFOrder,DiffRotOrder] +1,i++, time=AbsoluteTiming[rs=Append[rs,Expand[(r[l,\[Mu]])^i]/.OR^\[Alpha]_/;\[Alpha]>ToFOrder->0]];Print[N[i/Max[ToFOrder,DiffRotOrder]*100,3],"% complete. Step took ",time[[1]]," seconds."];];*)
(*Print["Ensuring that r and rfast are consistent..."]*)
(*For[i=1,i<Max[ToFOrder,DiffRotOrder] +1,i++,Print["This should be zero: ", FullSimplify[rfasts[[i]]-rs[[i]]]]];*)
(*rs = rfasts;*)*)


(* ::Subsubsection:: *)
(*Insert \[CapitalSigma][l,\[Mu]] and r[l,\[Mu]] up to relevant order*)


(* ::Input::Initialization:: *)
PowerLawfnz\[Mu][n_,z_,\[Mu]_]         =Expand[PowerLawfn[n,x]/.f[x,\[Beta]_]->Indexed[\[CapitalSigma]s,\[Beta]]/.x->\[CapitalSigma]s[[1]]]/.OR->1;(*equation (B.10)*)
PowerLawfdashnz\[Mu][n_,z_,\[Mu]_]=Expand[PowerLawfdashn[n,x]/.f[x,\[Beta]_]->Indexed[\[CapitalSigma]s,\[Beta]]/.x->\[CapitalSigma]s[[1]]]/.OR->1;(*equation (B.10)*)
PowerLawfdash2z\[Mu][z_,\[Mu]_]        =Expand[PowerLawfdash2[x]/.f[x,\[Beta]_]->Indexed[\[CapitalSigma]s,\[Beta]]/.x->\[CapitalSigma]s[[1]]]/.OR->1;(*equation (B.10)*)
PowerLawDnz\[Mu][n_,z_,\[Mu]_]         =Expand[PowerLawDn[n,x,l]/.f[x,\[Beta]_]->l^\[Beta] Indexed[\[CapitalSigma]s,\[Beta]]/.x->l \[CapitalSigma]s[[1]]] ;   (*equation (B.5)*)
PowerLawDdashnz\[Mu][n_,z_,\[Mu]_]=Expand[PowerLawDdashn[n,x,l]/.f[x,\[Beta]_]->l^\[Beta] Indexed[\[CapitalSigma]s,\[Beta]]/.x->l \[CapitalSigma]s[[1]]];(*equation (B.5)*)



(* ::Subsection:: *)
(*Define: Subscript[f, 2n],Subscript[f', 2n, ]V, Q, U, Subscript[A, 2n] *)


(* ::Input::Initialization:: *)
(*equation (B.10):*)
f[n_,z_]         :=3/(2(n+3))* Integrate[LegendreP[n,\[Mu]]*PowerLawfnz\[Mu][n,z,\[Mu]],{\[Mu],-1,1}]; fdash[n_,z_]:=3/(2(2-n))*Integrate[LegendreP[n,\[Mu]]*PowerLawfdashnz\[Mu][n,z,\[Mu]],{\[Mu],-1,1}];
fdash2[z_]       :=3/2*Integrate[LegendreP[2,\[Mu]]*PowerLawfdash2z\[Mu][z,\[Mu]],{\[Mu],-1,1}];
(*equations (B.5) and (B.7):*)
V[z_,\[Mu]_]=-G Sum[((PowerLawDnz\[Mu][2n,z,\[Mu]])*(4\[Pi] Subscript[\[Rho], av] (z*Rm)^(2n+3))/3 OR^n S[2n,z]+(PowerLawDdashnz\[Mu][2n,z,\[Mu]])*(4\[Pi] Subscript[\[Rho], av] (z*Rm)^(2-2n))/3 OR^n Sdash[2n,z])*LegendreP[2n,\[Mu]],{n,0,ToFOrder}];
(*equation (B.3):*)


(* ::Subsubsection:: *)
(*Definition of Q*)


(* ::Input::Initialization:: *)
(*differential rotation according to Zharkov et al. 1976, for solid body rotation set DiffRotOrder=0:*)
Q[z_,\[Mu]_]=-((m OR^1 G M)/Rm^3) (r[z*Rm,\[Mu]])^2 (1-\[Mu]^2)((1+Subscript[\[Alpha], 0])/2+Sum[(Subscript[\[Alpha], 2*i] OR^1)/(2*(i+1)Rm^(2*i)) rs[[2*i]](1-\[Mu]^2)^i,{i,1,DiffRotOrder/2}]);
Subscript[\[Alpha], 0]=If[DiffRotOrder==0,0,Subscript[\[Alpha], 0]];


(* ::Subsubsection:: *)
(*Definition of U*)


(* ::Input::Initialization:: *)
U[z_,\[Mu]_]=Expand[V[z,\[Mu]]+Q[z,\[Mu]]]/.OR^\[Alpha]_/;\[Alpha]>ToFOrder->0/.OR->1;
(*\[Integral]Subscript[P, n](x)Subscript[P, m](x)dx=2/(2n+1)Subscript[\[Delta], nm] when integrated from -1 to 1:*)
AA[n_,z_,\[Mu]_]=Expand[-(3/(4\[Pi])) (2*n+1)/2 1/(G Subscript[\[Rho], av] (z*Rm)^2) U[z,\[Mu]]*LegendreP[n,\[Mu]]];


(* ::Section:: *)
(*Calculate the coefficients Subscript[f, 2n] and Subscript[f', 2n]*)


(* ::Input::Initialization:: *)
label =List[];
For[i=0,i<ToFOrder +1,i++, label=Append[label,Subscript[f, 2*i]]]
labeldash=List[];
For[i=0,i<ToFOrder +1,i++, labeldash=Append[labeldash,Subscript[f', 2*i]]]


(* ::Subsection:: *)
(*Subscript[f, 2n]*)


(* ::Input::Initialization:: *)
fs = List[];
Print["Expanding fs..."]
For[i=0,i<ToFOrder+1,i++, 
time=AbsoluteTiming[fs=Append[fs,Expand[f[2*i,z]]/.s[\[Alpha]_,z Rm]->Subscript[s, \[Alpha]]]];
Print[N[(i+1)/(ToFOrder+1)*100,3],"% complete. Step took ",time[[1]]," seconds."];];
For[i=0,i<ToFOrder+1,i++,Print[label[[i+1]],"=",fs[[i+1]]]];


(* ::Input::Initialization:: *)
For[i=0,i<ToFOrder+1,i++,Print[label[[i+1]],"=",FortranForm[FullSimplify[fs[[i+1]]]/.Subscript[s, \[Alpha]_]->Indexed[ss,\[Alpha]+1]]]]
For[i=0,i<ToFOrder+1,i++,Export[FileNameJoin[{DirectoryName[$InputFileName],"f"<>ToString[2*i]<>".txt"}],FortranForm[FullSimplify[fs[[i+1]]]/.Subscript[s, \[Alpha]_]->Indexed[ss,\[Alpha]+1]],"Text"]];


(* ::Subsection:: *)
(*Subscript[f', 2n]*)


(* ::Input::Initialization:: *)
fdashs = List[];
Print["Expanding fdashs..."];
time=AbsoluteTiming[fdashs=Append[fdashs,Expand[fdash[0,z]]/.s[\[Alpha]_,z Rm]->Subscript[s, \[Alpha]]]];
Print[N[1/(ToFOrder+1)*100,3],"% complete. Step took ",time[[1]]," seconds."];
time=AbsoluteTiming[fdashs=Append[fdashs,Expand[fdash2[z]]/.s[\[Alpha]_,z Rm]->Subscript[s, \[Alpha]]]];
Print[N[2/(ToFOrder+1)*100,3],"% complete. Step took ",time[[1]]," seconds."];
For[i=2,i<ToFOrder+1,i++, 
time=AbsoluteTiming[fdashs=Append[fdashs,Expand[fdash[2*i,z]]/.s[\[Alpha]_,z Rm]->Subscript[s, \[Alpha]]]];
Print[N[(i+1)/(ToFOrder+1)*100,3],"% complete. Step took ",time[[1]]," seconds."];];
For[i=0,i<ToFOrder+1,i++,Print[labeldash[[i+1]],"=",fdashs[[i+1]]]];


(* ::Input::Initialization:: *)
For[i=0,i<ToFOrder+1,i++,Print[labeldash[[i+1]],"=",FortranForm[FullSimplify[fdashs[[i+1]]]/.Subscript[s, \[Alpha]_]->Indexed[ss,\[Alpha]+1]]]]
For[i=0,i<ToFOrder+1,i++,Export[FileNameJoin[{DirectoryName[$InputFileName],"f"<>ToString[2*i]<>"p.txt"}],FortranForm[FullSimplify[fdashs[[i+1]]]/.Subscript[s, \[Alpha]_]->Indexed[ss,\[Alpha]+1]],"Text"]];


(* ::Section:: *)
(*Calculate the Subscript[A, 2k]*)


(* ::Input::Initialization:: *)
labelA =List[];
labelA=Append[labelA,Subscript[A, 0]];
For[i=1,i<ToFOrder +1,i++, labelA=Append[labelA,Subscript[A, 2i]+Subscript[s, 2i] Subscript[S, 0]]]
As = List[];
Print["Expanding As..."]
For[i=0,i<ToFOrder+1,i++, 
time=AbsoluteTiming[As=Append[As,Integrate[AA[2*i,z,\[Mu]],{\[Mu],-1,1}]]];
Print[N[(i+1)/(ToFOrder+1)*100,3],"% complete. Step took ",time[[1]]," seconds."];];


(* ::Input::Initialization:: *)
Clear[l]
DisplayA[what_]:=Collect[Expand[what/.{m->3*m}],{S[0,z],S[1,z],S[2,z],S[3,z],S[4,z],S[5,z],S[6,z],S[7,z],S[8,z],Sdash[0,z],Sdash[1,z],Sdash[2,z],Sdash[3,z],Sdash[4,z],Sdash[5,z],Sdash[6,z],Sdash[7,z],Sdash[8,z],m}];
(*Subscript[A, 0]:*)
expression=DisplayA[As[[1]]]/.{s[\[Alpha]_,z Rm]->Subscript[s, \[Alpha]]}/.{S[\[Alpha]_,z]->Subscript[S, \[Alpha]]}/.{Sdash[\[Alpha]_,z]->Subscript[S', \[Alpha]]}/.{Rm^\[Alpha]_ z^\[Alpha]_->l^\[Alpha]};
(*Part without differential rotation:*)
Print[labelA[[1]],"=",expression/.{Subscript[\[Alpha], i_]->0}/.{m->m/3}];
(*Part with differential rotation:*)
Print["+",Collect[Expand[expression-(expression/.{Subscript[\[Alpha], i_]->0})],m]/.{m->m/3}];
(*Subscript[A, 2i]+Subscript[s, 2i]Subscript[S, 0] for i>0*)
For[i=1,i<ToFOrder+1,i++,expression=DisplayA[As[[i+1]]+s[2*i,z Rm]*S[0,z]]/.{s[\[Alpha]_,z Rm]->Subscript[s, \[Alpha]]}/.{S[\[Alpha]_,z]->Subscript[S, \[Alpha]]}/.{Sdash[\[Alpha]_,z]->Subscript[S', \[Alpha]]}/.{Rm^\[Alpha]_ z^\[Alpha]_->l^\[Alpha]};
(*Part without differential rotation:*)
Print[labelA[[i+1]],"=",expression/.{Subscript[\[Alpha], i_]->0}/.{m->m/3}];
(*Part with differential rotation:*)
Print["+",Collect[Expand[expression-(expression/.{Subscript[\[Alpha], i_]->0})],m]/.{m->m/3}]];


(* ::Input::Initialization:: *)
(*Subscript[A, 0]:*)
expression=DisplayA[As[[1]]]/.s[\[Alpha]_,z Rm]->Indexed[ss,\[Alpha]+1]/.S[\[Alpha]_,z ]->Indexed[SS,\[Alpha]+1]/.Sdash[\[Alpha]_,z ]->Indexed[SSdash,\[Alpha]+1];
(*Part without differential rotation:*)
Print[labelA[[1]],"=",FortranForm[Simplify[expression]/.{Subscript[\[Alpha], i_]->0}/.{m->m/3}]];
Export[FileNameJoin[{DirectoryName[$InputFileName],"A0_no_DR.txt"}],FortranForm[Simplify[expression]/.{Subscript[\[Alpha], i_]->0}/.{m->m/3}],"Text"];
(*Part with differential rotation:*)
Print["+",FortranForm[Simplify[Collect[Expand[expression-(expression/.{Subscript[\[Alpha], i_]->0})],m]]/.Subscript[\[Alpha], \[Beta]_]->Indexed[alphas,\[Beta]+1]/.{m->m/3}]];
Export[FileNameJoin[{DirectoryName[$InputFileName],"A0_only_DR.txt"}],FortranForm[Simplify[Collect[Expand[expression-(expression/.{Subscript[\[Alpha], i_]->0})],m]]/.Subscript[\[Alpha], \[Beta]_]->Indexed[alphas,\[Beta]+1]/.{m->m/3}],"Text"];
(*Subscript[A, 2i]+Subscript[s, 2i]Subscript[S, 0] for i>0*)
For[i=1,i<ToFOrder+1,i++,expression=DisplayA[As[[i+1]]+s[2*i,z Rm]*S[0,z]]/.s[\[Alpha]_,z Rm]->Indexed[ss,\[Alpha]+1]/.S[\[Alpha]_,z ]->Indexed[SS,\[Alpha]+1]/.Sdash[\[Alpha]_,z ]->Indexed[SSdash,\[Alpha]+1];
(*Part without differential rotation:*)
Print[labelA[[i+1]],"=",FortranForm[Simplify[expression]/.{Subscript[\[Alpha], i_]->0}/.{m->m/3}]];
Export[FileNameJoin[{DirectoryName[$InputFileName],"A"<>ToString[2*i]<>"+s"<>ToString[2*i]<>"S0_no_DR.txt"}],FortranForm[Simplify[expression]/.{Subscript[\[Alpha], i_]->0}/.{m->m/3}],"Text"];
(*Part with differential rotation:*)
Print["+",FortranForm[Simplify[Collect[Expand[expression-(expression/.{Subscript[\[Alpha], i_]->0})],m]]/.Subscript[\[Alpha], \[Beta]_]->Indexed[alphas,\[Beta]+1]/.{m->m/3}]];
Export[FileNameJoin[{DirectoryName[$InputFileName],"A"<>ToString[2*i]<>"+s"<>ToString[2*i]<>"S0_only_DR.txt"}],FortranForm[Simplify[Collect[Expand[expression-(expression/.{Subscript[\[Alpha], i_]->0})],m]]/.Subscript[\[Alpha], \[Beta]_]->Indexed[alphas,\[Beta]+1]/.{m->m/3}],"Text"]];


(* ::Section:: *)
(*Quit the Kernel*)


(* ::Input::Initialization:: *)
Quit[];
