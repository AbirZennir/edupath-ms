package com.example.controllers;

import com.example.dto.RiskPredictionResponse;
import com.example.services.RiskPredictionService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/students")
@RequiredArgsConstructor
public class RiskPredictionController {

    private final RiskPredictionService riskPredictionService;

    /**
     * Calcule les features de l'étudiant, appelle le microservice IA et persiste la prédiction.
     * codePresentation est optionnel (si présent, filtre les données VLE/assessments correspondantes).
     */
    @PostMapping(value = "/{id}/predict-risk", produces = MediaType.APPLICATION_JSON_VALUE)
    public RiskPredictionResponse predict(@PathVariable("id") int studentId,
                                          @RequestParam(name = "codePresentation", required = false) String codePresentation) {
        return riskPredictionService.predict(studentId, codePresentation);
    }
}
