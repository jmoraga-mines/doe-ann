# Scripts to load images

if (!require("pacman")){
  install.packages("pacman")
  require("pacman")
}
# pacman::p_load(raster, tools, rasterKernelEstimates, spdep)
pacman::p_load(raster)

extent_brady <- raster(xmn=327499.1, xmx=329062.1, ymn = 4405906, ymx=4409320, res=c(3,3), crs="+proj=utm +zone=11 +datum=WGS84 +units=m +no_defs +ellps=WGS84 +towgs84=0,0,0")
# Larger extent, we need to resample the Geothermal, Temperature and Fault images
# extent_brady <- raster(xmn=327385.1, xmx=329149.1, ymn = 4405876, ymx=4409353, res=c(3,3), crs="+proj=utm +zone=11 +datum=WGS84 +units=m +no_defs +ellps=WGS84 +towgs84=0,0,0")
south_brady <- raster(xmn=327511.1, xmx=328030.1, ymn = 4405945, ymx= 4406430, res=c(3,3), crs="+proj=utm +zone=11 +datum=WGS84 +units=m +no_defs +ellps=WGS84 +towgs84=0,0,0")

doe_writeRaster <- function(x, filename, format="raster", overwrite=TRUE, bandorder="BSQ"){
  if(tools::file_ext(filename) != "grd") {
    filename <- tools::file_path_sans_ext(filename)
    filename <- paste(filename, ".grd", sep="")
  }
  f1<-writeRaster(x=x, filename=filename, bandorder=bandorder, 
                  format=format, overwrite=overwrite)
  hdr(f1, "ENVI")
  return(f1)
}

brady <- stack("/store03/geodata/MachineLearning/doe-som/brady_som_output.grd")
crs(brady) <- crs('+init=epsg:32611')
f <- doe_writeRaster( brady, filename="/store03/geodata/MachineLearning/rasters/brady.grd")
rm(f)
dem_brady <- stack("/store03/geodata/DEM_Asp_Slope/brady_DEM.grd")
dem_brady <- projectRaster(dem_brady, brady, crs=crs(b))
names(dem_brady) <- "Elevation"
brady_01 <- stack(brady, dem_brady)

brady_slope <- stack('/store03/geodata/DEM_Asp_Slope/brady_slope.grd')
brady_slope <- projectRaster(brady_slope, brady, crs=crs('+init=epsg:32611'))
# crs(brady_slope) <- crs('+init=epsg:32611')
names(brady_slope) <- "Slope"
brady_02 <- stack(brady, brady_slope)

brady_aspect <- stack('/store03/geodata/DEM_Asp_Slope/brady_aspect.grd')
brady_aspect <- projectRaster(brady_aspect, brady, crs=crs('+init=epsg:32611'))
# crs(brady_aspect) <- crs('+init=epsg:32611')
names(brady_aspect) <- "Aspect"
brady_03 <- stack(brady, brady_slope, brady_aspect)
crs(brady_01) <- crs('+init=epsg:32611')
crs(brady_02) <- crs('+init=epsg:32611')
crs(brady_03) <- crs('+init=epsg:32611')

f1 <- doe_writeRaster( brady_01, filename="/store03/geodata/MachineLearning/rasters/brady_01.grd")
f2 <- doe_writeRaster( brady_02, filename="/store03/geodata/MachineLearning/rasters/brady_02.grd")
f3 <- doe_writeRaster( brady_03, filename="/store03/geodata/MachineLearning/rasters/brady_03.grd")

rm(f1, f2, f3)

