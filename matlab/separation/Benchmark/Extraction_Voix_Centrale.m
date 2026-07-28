function [Signal_Sortie,Nombre_Fenetre] = Extraction_Voix_Centrale(Signal_Entree,Fs,Type)

Nombre_Canaux_Entree = size(Signal_Entree,2);
Nombre_Canaux_Sortie = Nombre_Canaux_Entree + 1;

Taille_Buffer = 2^(nextpow2(Fs/100));
Taille_Fenetre = 2 * Taille_Buffer;

Coefficient_Smooth = 0.05;
Coefficient_Smooth = (1 - exp((-3 * Taille_Buffer)./(Fs * Coefficient_Smooth)));

Window = sqrt( 1 - 0.5*(1 + (cos(pi*(1:Taille_Fenetre)'/((Taille_Fenetre)/2)))) );

[RF_Fenetre,Nombre_Fenetre] = STFFT(Signal_Entree,Taille_Buffer,Window,1);

Spectre_Energie_Precedent = zeros(Taille_Buffer + 1,Nombre_Canaux_Entree);

if strcmp(Type,'RF')
    Signal_Sortie = zeros(Taille_Fenetre/2 + 1,Nombre_Canaux_Sortie,Nombre_Fenetre);
elseif strcmp(Type,'Temporel')
    Signal_Sortie = zeros((Nombre_Fenetre - 1) * Taille_Buffer,Nombre_Canaux_Sortie);
end

Signal_Fenetre_Canal = zeros(Taille_Fenetre,Nombre_Canaux_Sortie);

for fenetre = 1:Nombre_Fenetre
    
        
        [Masque,Somme,Spectre_Energie_Precedent] = Extraction_VC_Calcul_Masque(RF_Fenetre(:,1,fenetre),RF_Fenetre(:,2,fenetre),Coefficient_Smooth,Spectre_Energie_Precedent);
        
        RF_Canal(:,1) = Masque .* Somme;
        RF_Canal(:,2) = RF_Fenetre(:,1,fenetre) - RF_Canal(:,1) ./ sqrt(2);
        RF_Canal(:,3) = RF_Fenetre(:,2,fenetre) - RF_Canal(:,1) ./ sqrt(2);

        RF_Canal = circshift(RF_Canal,[0 -1]);

        if Nombre_Canaux_Entree > 2
            RF_Canal(:,4) = RF_Fenetre(:,3,fenetre);
            RF_Canal(:,5) = RF_Fenetre(:,4,fenetre);
        end
        
        
        if strcmp(Type,'RF')
            
            Signal_Sortie(:,:,fenetre) = RF_Canal;
            
        elseif strcmp(Type,'Temporel')
            
            for i = 1:3
                Signal_Fenetre_Canal(:,i) = Extraction_VC_Ifft(RF_Canal(:,i),Taille_Fenetre,Window);
            end
            
            if fenetre > 1
                Signal_Sortie_Fenetre(:,:) = Signal_Precedente_Fenetre_Canal(:,:) + Signal_Fenetre_Canal(1 : Taille_Buffer,:);
                Signal_Sortie( 1 + (fenetre - 2) * Taille_Buffer : (fenetre - 1) * Taille_Buffer,:)  = Signal_Sortie_Fenetre;
            end
            
            Signal_Precedente_Fenetre_Canal(:,:) =  Signal_Fenetre_Canal(Taille_Buffer + 1 : Taille_Fenetre,:) ;
        
        end

end