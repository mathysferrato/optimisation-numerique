@doc doc"""
#### Objet
Cette fonction calcule une solution approchée du problème

```math
\min_{||s||< \Delta}  q(s) = s^{t} g + \frac{1}{2} s^{t}Hs
```

par l'algorithme du gradient conjugué tronqué

#### Syntaxe
```julia
s = Gradient_Conjugue_Tronque(g,H,option)
```

#### Entrées :   
   - g : (Array{Float,1}) un vecteur de ``\mathbb{R}^n``
   - H : (Array{Float,2}) une matrice symétrique de ``\mathbb{R}^{n\times n}``
   - options          : (Array{Float,1})
      - delta    : le rayon de la région de confiance
      - max_iter : le nombre maximal d'iterations
      - tol      : la tolérance pour la condition d'arrêt sur le gradient

#### Sorties:
   - s : (Array{Float,1}) le pas s qui approche la solution du problème : ``min_{||s||< \Delta} q(s)``

#### Exemple d'appel:
```julia
gradf(x)=[-400*x[1]*(x[2]-x[1]^2)-2*(1-x[1]) ; 200*(x[2]-x[1]^2)]
hessf(x)=[-400*(x[2]-3*x[1]^2)+2  -400*x[1];-400*x[1]  200]
xk = [1; 0]
options = []
s = Gradient_Conjugue_Tronque(gradf(xk),hessf(xk),options)
```
"""
function Gradient_Conjugue_Tronque(g,H,options)

    "# Si option est vide on initialise les 3 paramètres par défaut"
    if options == []
        delta = 2
        max_iter = 100
        tol = 1e-6
    else
        delta = options[1]
        max_iter = options[2]
        tol = options[3]
    end

    n = length(g)
    s = zeros(n)

    j = 0
    g0 = g
    p = -g
    while (j < 2*n) && (norm(g) > max(norm(g0)*tol, tol))
        k = p'*H*p
        if (k <= 0)
            a = norm(p)^2
            b = 2*p'*s
            c = norm(s)^2 - delta^2
            discr = b^2 - 4*a*c
            sigma1 = (-b-sqrt(discr))/(2*a)
            sigma2 = (-b+sqrt(discr))/(2*a)
            q1 = g'*(s + sigma1*p) + (1/2)*(s + sigma1*p)'*H*(s + sigma1*p)
            q2 = g'*(s + sigma2*p) + (1/2)*(s + sigma2*p)'*H*(s + sigma2*p)
            q = min(q1,q2)
            if (q == q1)
                sigma = sigma1
            else
                sigma = sigma2
            end
            return s + sigma*p
        end
        alpha = g'*g/k
        if (norm(s + alpha*p) >= delta)
            a = norm(p)^2
            b = 2*p'*s
            c = norm(s)^2 - delta^2
            discr = b^2 - 4*a*c
            sigma = (-b+sqrt(discr))/(2*a)
            return s + sigma*p
        end
        gj = g

        s = s + alpha*p
        g = g + alpha*H*p
        beta = g'*g/(gj'*gj)
        p = -g + beta*p
        j = j + 1
    end
   return s
end
