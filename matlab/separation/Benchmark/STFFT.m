function [RF_Fenetre,Nombre_Fenetre] = STFFT(Signal_Entree,Taille_Buffer,Window,Shanon)


% -- Calcul des paramètres --

% On calcule la longueur de notre signal ainsi que de le nombre de canaux
Nombre_Echantillon = length(Signal_Entree);
Nombre_Canaux = size(Signal_Entree,2);

% On calcule le nombre de buffer de taille Taille_Buffer que l'on a dans notre signal
Nombre_Buffer = ceil(Nombre_Echantillon / Taille_Buffer);

% On calcule le nombre de fenêtre (1 fenêtre = double buffer) que l'on doit avoir
Nombre_Fenetre = Nombre_Buffer + 1;


% -- Initialisation de notre signal de travail (Signal d'entrée complété de zéros avant le premier buffer, pour le dernier buffer (si besoin), et après le dernier buffer) --

% On remplit de 0 notre signal de travail
Signal = zeros((Nombre_Buffer + 2) * Taille_Buffer,Nombre_Canaux);

% On ajoute notre signal d'entrée au bonne endroit
Signal (Taille_Buffer + 1 : Taille_Buffer + Nombre_Echantillon,:) = Signal_Entree;


% -- Initialisation des différentes matrices de travail --

Signal_Fenetre = zeros(2 * Taille_Buffer,Nombre_Canaux,Nombre_Fenetre);

if Shanon == 1
    RF_Fenetre = zeros(Taille_Buffer + 1,Nombre_Canaux,Nombre_Fenetre);
else
    RF_Fenetre = zeros(2 * Taille_Buffer,Nombre_Canaux,Nombre_Fenetre);
end


% -- Récupération des différentes fenêtres et calcul de chaque RF --

Window = repmat(Window,1,Nombre_Canaux);

for fenetre = 1 : Nombre_Buffer + 1

    % Récupération de la fenêtre et fenêtrage
    Signal_Fenetre(:,:,fenetre) = Signal( 1 + (fenetre - 1) * Taille_Buffer : (fenetre + 1) * Taille_Buffer,:) .* Window;
        
end

for canal = 1 : Nombre_Canaux
    
    % Calcul de la réponse en fréquence de la fenêtre
    RF_Total_Fenetre_Canal = fft(squeeze(Signal_Fenetre(:,canal,:)));
    
    % Récupération de la partie utile (avant fréquence de Shanon) de la réponse en fréquence de la fenêtre
    if Shanon == 1
        RF_Fenetre(:,canal,:) = RF_Total_Fenetre_Canal (1:Taille_Buffer + 1,:);
    else
        RF_Fenetre(:,canal,:) = RF_Total_Fenetre_Canal (:,:);
    end
    
end