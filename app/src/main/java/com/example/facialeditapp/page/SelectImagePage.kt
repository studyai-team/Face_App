package com.example.facialeditapp.page

import android.content.Intent
import android.os.Bundle
import android.view.View
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.Toolbar
import com.example.facialeditapp.R


class SelectImagePage : AppCompatActivity() {
    private val PICK_IMAGE = 100
    companion object {
        val IMAGE_URI = "IMAGE"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.select_image_page)
        val myToolbar: Toolbar = findViewById(R.id.my_toolbar)
        setSupportActionBar(myToolbar)
    }

    public fun clickUploadButton(view: View) {
        openGallery();
    }

    private fun openGallery() {
        val intent = Intent()
        intent.type = "image/*"
        intent.action = Intent.ACTION_GET_CONTENT
        startActivityForResult(Intent.createChooser(intent, "Select Picture"), PICK_IMAGE)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == PICK_IMAGE) {
            val imageUri = data?.data
            val intent = Intent(this@SelectImagePage, ChooseTypePage::class.java)
            intent.putExtra(IMAGE_URI, imageUri)
            startActivity(intent)
        }
    }
}