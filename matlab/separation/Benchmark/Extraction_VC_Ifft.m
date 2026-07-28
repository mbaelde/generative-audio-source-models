function Signal_Sortie = Extraction_VC_Ifft(RF_Entree,Taille_Fenetre,Window)

RF_Entree (Taille_Fenetre/2 + 2 : Taille_Fenetre) = conj(RF_Entree (Taille_Fenetre / 2 : -1 : 2));

Signal_Sortie = real(ifft(RF_Entree)) .* Window;