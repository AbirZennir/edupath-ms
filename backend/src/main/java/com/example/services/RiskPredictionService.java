package com.example.services;

import com.example.dto.RiskPredictionResponse;
import com.example.entities.StudentRiskPrediction;
import com.example.entities.learning.StudentInfo;
import com.example.repositories.*;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.time.Instant;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

@Service
@RequiredArgsConstructor
public class RiskPredictionService {

    private final StudentInfoRepository studentInfoRepository;
    private final StudentVleRepository studentVleRepository;
    private final StudentAssessmentRepository studentAssessmentRepository;
    private final StudentRiskPredictionRepository studentRiskPredictionRepository;
    private final RestTemplate restTemplate;

    @Value("${ai.risk.url:http://localhost:8001/predict-risk}")
    private String aiRiskUrl;

    /**
     * Calcule les features minimales en se basant sur les données existantes.
     * Certaines features sont approchées faute de détails (eng_clicks_per_day, assess_per_10days).
     */
    private Map<String, Object> buildFeatures(int studentId, String codePresentation) {
        Optional<StudentInfo> infoOpt = studentInfoRepository.findById(studentId);

        long sumClickTotal = studentVleRepository.sumClicksByStudent(studentId, codePresentation);
        long nAssessments = studentAssessmentRepository.countByStudent(studentId, codePresentation);

        // Approximations faute de granularité :
        double engClicksPerDay = sumClickTotal / 30.0;   // suppose 30 jours d'observation
        double assessPer10Days = nAssessments / 3.0;     // suppose ~30 jours => 3 fenêtres de 10 jours

        double studiedCredits = infoOpt.map(StudentInfo::getStudied_credits).orElse(0);
        double numPrevAttempts = 0;
        try {
            numPrevAttempts = infoOpt.map(StudentInfo::getNum_of_prev_attempts)
                    .map(v -> {
                        try {
                            return Double.parseDouble(v);
                        } catch (NumberFormatException e) {
                            return 0.0;
                        }
                    }).orElse(0.0);
        } catch (Exception e) {
            numPrevAttempts = 0.0;
        }

        Map<String, Object> payload = new HashMap<>();
        payload.put("sum_click_total", (double) sumClickTotal);
        payload.put("n_assessments", (double) nAssessments);
        payload.put("eng_clicks_per_day", engClicksPerDay);
        payload.put("assess_per_10days", assessPer10Days);
        payload.put("studied_credits", studiedCredits);
        payload.put("num_of_prev_attempts", numPrevAttempts);
        return payload;
    }

    public RiskPredictionResponse predict(int studentId, String codePresentation) {
        Map<String, Object> features = buildFeatures(studentId, codePresentation);

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        HttpEntity<Map<String, Object>> request = new HttpEntity<>(features, headers);

        @SuppressWarnings("unchecked")
        Map<String, Object> resp = restTemplate.postForObject(aiRiskUrl, request, Map.class);
        if (resp == null || !resp.containsKey("risk_score")) {
            throw new RuntimeException("Réponse IA invalide");
        }

        double riskScore = ((Number) resp.get("risk_score")).doubleValue();
        int prediction = ((Number) resp.get("prediction")).intValue();
        String modelType = (String) resp.getOrDefault("model_type", "mlp");

        StudentRiskPrediction entity = new StudentRiskPrediction();
        entity.setStudentId(studentId);
        entity.setCodePresentation(codePresentation);
        entity.setRiskScore(riskScore);
        entity.setPrediction(prediction);
        entity.setModelType(modelType);
        entity.setCreatedAt(Instant.now());
        studentRiskPredictionRepository.save(entity);

        return new RiskPredictionResponse(studentId, codePresentation, riskScore, prediction, modelType);
    }
}
