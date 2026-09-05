function [F,A,g,J,Jn,U0,h]=Adr_2d_function(epsilon,alpha,gamma,n)
    h=1/n;
    x = linspace(0,1,n+1);
    y = linspace(0,1,n+1);
    [X,Y] = meshgrid(x,y);
    X = X';
    Y = Y';
    initial_condition=0.3+256*((X.*(1-X)).*(Y.*(1-Y))).*((X.*(1-X)).*(Y.*(1-Y)));
    U0=reshape(initial_condition,(n+1)*(n+1),1);
    I = speye(n+1);
    e = ones(n+1,1); 
    %Define Ad
    Ad = spdiags([e -4*e e],[-1 0 1],n+1,n+1);
    Ad(1,2)=2;
    Ad(end,end-1)=2; 
    Sd = spdiags([e e],[-1 1],n+1,n+1);
    Sd(1,2)=2;
    Sd(end,end-1)=2; 
    %Define Aa
    Aa = spdiags([-e 0*e e],[-1 0 1],n+1,n+1);
    Aa(1,2)=0;
    Aa(end,end-1)=0;
    Sa = spdiags([-e e],[-1 1],n+1,n+1);
    Sa(1,2)=0;
    Sa(end,end-1)=0;
    Ad = (kron(I,Ad) + kron(Sd,I));
    Aa = (kron(I,Aa) + kron(Sa,I));
    A= (epsilon/h^2)*Ad-(alpha/(2*h))*Aa;
    %Define Jacobian
    g=@(u) gamma*u.*(u-1/2).*(1-u);
    deriv_g=@(u)gamma*(-3*u.^2+3*u-1/2);
    Jn =@(u)A+spdiags(deriv_g(u),[0],(n+1)^2,(n+1)^2);
    J =@(u) spdiags(gamma*(-3*u.^2+3*u-1/2),[0],(n+1)^2,(n+1)^2);
    F=@(u)A*u+g(u);
end