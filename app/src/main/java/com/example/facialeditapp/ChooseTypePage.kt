package com.example.facialeditapp

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.view.View
import android.widget.ImageView
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.Toolbar

class ChooseTypePage : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.choose_type_page)
        val myToolbar: Toolbar = findViewById(R.id.my_toolbar)
        setSupportActionBar(myToolbar)

        val imageUri = intent.extras?.get(SelectImagePage.IMAGE_URI) as Uri

        val imageView = findViewById<ImageView>(R.id.selectedImage)
        imageView.setImageURI(imageUri)
    }

    public fun clickEditButton(view: View) {
        val intent = Intent(this@ChooseTypePage, SaveImagePage::class.java)
        startActivity(intent)
    }
}