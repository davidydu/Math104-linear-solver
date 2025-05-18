function [x,meta] = linear_solver(A,b,tol)
%LINEAR_SOLVER  Solve Ax = b using the textbook rank–case catalogue.
%
%   [x,meta] = LINEAR_SOLVER(A,b) decides automatically among the four
%   canonical situations discussed in most linear–algebra / numerical–
%   optimization texts (see lecture‑note cases 1(i), 1(ii), 2(i), 2(ii)):
%
%     Case 1(i)  (square, full rank)          – unique solution  x = A\\b.
%     Case 1(ii) ("wide", m < n, full row rank)
%               – **minimal‑norm solution**  x = A^Rb = A'*(A*A')^{-1}b.
%     Case 2(i)  ("tall", m > n, full column rank)
%               – **least‑squares solution** (unique)  x = (A'A)^{-1}A'b.
%     Case 2(ii) (rank‑deficient)
%               – If consistent (b \in R(A)):
%                     **minimal‑norm solution**  x = A^+b.
%               – If inconsistent (b ∉ R(A)):
%                     **least‑squares solution of minimum norm**  x = A^+b.
%
%   Here A^R, A^L, and A^+ denote the right inverse, left inverse, and
%   Moore–Penrose pseudoinverse, respectively.  We implement them with
%   basic MATLAB primitives so the script is toolbox‑free.
%
%   meta – diagnostic struct
%     meta.rank            numerical rank of A (within tolerance)
%     meta.case            'square' | 'wide' | 'tall' | 'rank‑deficient'
%     meta.solution_type   textbook label shown above
%     meta.residual        2‑norm of A*x − b
%     meta.norm_x          2‑norm of x
%
%   Example
%     A = randn(3,5);  b = randn(3,1);
%     [x,meta] = linear_solver(A,b);
%     fprintf('Rank = %d, %s\n',meta.rank,meta.solution_type)
%
%   See also PINV, LSQMINNORM.

    if nargin < 3
        tol = max(size(A)) * eps(norm(A,'fro'));
    end

    [m,n] = size(A);
    r = rank(A,tol);

    meta.rank = r;
    meta.tol  = tol;

    % Orthogonal projector onto R(A)⊥ to test consistency using same tolerance
    consistent = norm((eye(m) - A*pinv(A, tol))*b) < tol;

    if m == n && r == n                 % Case 1(i)
        meta.case = 'square';
        x = A\b;
        meta.solution_type = 'unique solution';

    elseif r == m && m < n             % Case 1(ii) – wide, full row rank
        meta.case = 'wide';
        x = A' * ((A*A')\b);           % right inverse A^R b
        meta.solution_type = 'minimal‑norm solution';

    elseif r == n && n < m             % Case 2(i) – tall, full column rank
        meta.case = 'tall';
        x = (A'*A)\(A'*b);             % left inverse A^L b
        meta.solution_type = 'least‑squares solution (unique)';

    else                               % Case 2(ii) – rank deficient
        meta.case = 'rank‑deficient';
        x = pinv(A, tol)*b;            % Moore–Penrose pseudoinverse using same tolerance
        if consistent
            meta.solution_type = 'minimal‑norm solution (consistent)';
        else
            meta.solution_type = 'least‑squares solution of minimum norm';
        end
    end

    meta.residual = norm(A*x - b);
    meta.norm_x   = norm(x);
end
