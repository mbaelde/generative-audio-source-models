function [Masque,S,Spectre_Energie_Precedent] = Extraction_VC_Calcul_Masque(RF_Gauche,RF_Droite,Coefficient_Smooth,Spectre_Energie_Precedent)

S = (RF_Gauche + RF_Droite);
D = (RF_Gauche - RF_Droite);

S_Norme = abs(S) .^2;
D_Norme = abs(D) .^2;

S_Norme_Smooth = (1 - Coefficient_Smooth) .* Spectre_Energie_Precedent(:,1) + Coefficient_Smooth .* S_Norme;
D_Norme_Smooth = (1 - Coefficient_Smooth) .* Spectre_Energie_Precedent(:,2) + Coefficient_Smooth .* D_Norme;

Spectre_Energie_Precedent(:,1) = S_Norme_Smooth;
Spectre_Energie_Precedent(:,2) = D_Norme_Smooth;

Masque = ( 1 - sqrt( D_Norme_Smooth ./ (S_Norme_Smooth + 0.000001) ) ) ./ sqrt(2) ;