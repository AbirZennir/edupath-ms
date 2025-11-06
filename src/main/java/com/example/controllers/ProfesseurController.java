package com.example.controllers;

import com.example.entities.Professeur;
import com.example.repositories.ProfesseurRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("eduPath")
public class ProfesseurController {
    @Autowired
    private ProfesseurRepository professeurRepository;

    @GetMapping("/professeurs")
    public List<Professeur> findAll() {
        return professeurRepository.findAll();
    }

    @GetMapping("/professeurs/{id}")
    public ResponseEntity<Professeur> findById(@PathVariable Integer id) {
        return professeurRepository.findById(id)
                .map(ResponseEntity::ok)
                .orElse(ResponseEntity.notFound().build());
    }

    @PostMapping("/professeurs")
    public Professeur create(@RequestBody Professeur professeur) {
        return professeurRepository.save(professeur);
    }

    @PutMapping("/professeurs/{id}")
    public ResponseEntity<Professeur> update(@PathVariable Integer id, @RequestBody Professeur professeurDetails) {
        return professeurRepository.findById(id)
                .map(professeur -> {
                    professeur.setNom(professeurDetails.getNom());
                    professeur.setPrenom(professeurDetails.getPrenom());
                    professeur.setEmail(professeurDetails.getEmail());
                    professeur.setPassword(professeurDetails.getPassword());
                    professeur.setAdresse(professeurDetails.getAdresse());
                    professeur.setTelephone(professeurDetails.getTelephone());
                    professeur.setGenre(professeurDetails.getGenre());
                    Professeur updated = professeurRepository.save(professeur);
                    return ResponseEntity.ok(updated);
                })
                .orElse(ResponseEntity.notFound().build());
    }

    @DeleteMapping("/professeurs/{id}")
    public ResponseEntity<Void> delete(@PathVariable Integer id) {
        return professeurRepository.findById(id)
                .map(professeur -> {
                    professeurRepository.delete(professeur);
                    return ResponseEntity.noContent().<Void>build();
                })
                .orElse(ResponseEntity.notFound().build());
    }
}

