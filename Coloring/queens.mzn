include "alldifferent.mzn";

int: n;
set of int : R = 1..n;
set of int : C = 1..n;
array[C] of var R: row;
array[R] of var C: col;

constraint alldifferent(row);
constraint alldifferent([row[i] + i| i in R]);
constraint alldifferent([row[i] - i| i in R]);

constraint alldifferent(col);
constraint alldifferent([col[i] + i| i in C]);
constraint alldifferent([col[i] - i| i in C]);

constraint forall(r in R, c in C)(row[c] == r <-> col[r] == c);

% solve :: int_search(row, first_fail, indomain_min)
%       satisfy;
solve satisfy;

% output [ if fix(row[j]) == i then "* " else "0 " endif ++
%          if j == n then "\n" else "" endif | i,j in R]