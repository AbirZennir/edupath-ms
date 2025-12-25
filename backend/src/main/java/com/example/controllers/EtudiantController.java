package com.example.controllers;

import com.example.entities.Etudiant;
import com.example.repositories.EtudiantRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api")
public class EtudiantController {
    @Autowired
    private EtudiantRepository etudiantRepository;

    @GetMapping("/etudiants")
    public List<Etudiant> findAll() {
        return etudiantRepository.findAll();
    }

    @GetMapping("/etudiants/{id}")
    public ResponseEntity<Etudiant> findById(@PathVariable Integer id) {
        return etudiantRepository.findById(id)
                .map(ResponseEntity::ok)
                .orElse(ResponseEntity.notFound().build());
    }

    @PostMapping("/etudiants")
    public Etudiant create(@RequestBody Etudiant etudiant) {
        return etudiantRepository.save(etudiant);
    }

    @PutMapping("/etudiants/{id}")
    public ResponseEntity<Etudiant> update(@PathVariable Integer id, @RequestBody Etudiant etudiantDetails) {
        return etudiantRepository.findById(id)
                .map(etudiant -> {
                    etudiant.setNom(etudiantDetails.getNom());
                    etudiant.setPrenom(etudiantDetails.getPrenom());
                    etudiant.setEmail(etudiantDetails.getEmail());
                    etudiant.setPassword(etudiantDetails.getPassword());
                    etudiant.setAdresse(etudiantDetails.getAdresse());
                    etudiant.setTelephone(etudiantDetails.getTelephone());
                    etudiant.setGenre(etudiantDetails.getGenre());
                    Etudiant updatedEtudiant = etudiantRepository.save(etudiant);
                    return ResponseEntity.ok(updatedEtudiant);
                })
                .orElse(ResponseEntity.notFound().build());
    }

    @DeleteMapping("/etudiants/{id}")
    public ResponseEntity<Void> delete(@PathVariable Integer id) {
        return etudiantRepository.findById(id)
                .map(etudiant -> {
                    etudiantRepository.delete(etudiant);
                    return ResponseEntity.noContent().<Void>build();
                })
                .orElse(ResponseEntity.notFound().build());
    }
}
