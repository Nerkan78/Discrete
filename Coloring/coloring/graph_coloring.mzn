% include "alldifferent.mzn";
include "data.dzn";
include "globals.mzn";
int: N_Vertices;
int: N_Edges;
int: MinColor;
int: MaxColor;
array[1..N_Edges] of int: Adjacency_First_Node;
array[1..N_Edges] of int: Adjacency_Second_Node;
% Adjacency = [| 1, 2 | 2, 3 | 2, 4|];
array [1..N_Vertices] of var 1..MaxColor: coloring;




constraint max(coloring) > MinColor;
constraint forall(e in 1..N_Edges)(coloring[Adjacency_First_Node[e]] != coloring[Adjacency_Second_Node[e]]);
constraint symmetry_breaking_constraint(coloring[1] == 1);
constraint symmetry_breaking_constraint(seq_precede_chain(coloring));

% solve satisfy;
solve :: int_search(coloring, first_fail, indomain_min) minimize (max(coloring));
output ["\(fix(coloring[i]-1)) " | i in 1..N_Vertices] ++ ["\n\(max(fix(coloring)))"]