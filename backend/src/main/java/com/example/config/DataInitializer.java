package com.example.config;

import com.example.dto.RegisterRequest;
import com.example.services.UtilisateurService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.CommandLineRunner;
import org.springframework.stereotype.Component;

import java.util.List;

@Component
public class DataInitializer implements CommandLineRunner {
    private static final Logger LOGGER = LoggerFactory.getLogger(DataInitializer.class);

    private final UtilisateurService utilisateurService;

    public DataInitializer(UtilisateurService utilisateurService) {
        this.utilisateurService = utilisateurService;
    }

    @Override
    public void run(String... args) {
        List<RegisterRequest> seeds = List.of(
                buildRequest("Amine", "Azzam", "enseignant@edupath.fr", "EduPath123!", "EduPath Institute"),
                buildRequest("Samuel", "Admin", "admin@edupath.fr", "EduPath123!", "EduPath Institute"),
                buildRequest("Elazzam", "Lam", "elazzamilham2@gmail.com", "EduPath123!", "EduPath Institute")
        );

        seeds.forEach(this::ensureAccount);
    }

    private RegisterRequest buildRequest(String prenom, String nom, String email, String password, String institution) {
        RegisterRequest request = new RegisterRequest();
        request.setPrenom(prenom);
        request.setNom(nom);
        request.setEmail(email);
        request.setPassword(password);
        request.setInstitution(institution);
        request.setRole("PROFESSEUR");
        return request;
    }

    private void ensureAccount(RegisterRequest request) {
        if (utilisateurService.findByEmail(request.getEmail()) == null) {
            utilisateurService.register(request);
            LOGGER.info("Created default user {}", request.getEmail());
        }
    }
}
