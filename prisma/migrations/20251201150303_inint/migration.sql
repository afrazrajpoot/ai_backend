-- AlterTable
ALTER TABLE "User" ADD COLUMN     "paid" BOOLEAN NOT NULL DEFAULT false,
ADD COLUMN     "resetToken" TEXT,
ADD COLUMN     "resetTokenExpiry" TIMESTAMP(3),
ADD COLUMN     "verificationToken" TEXT;

-- CreateTable
CREATE TABLE "AnalysisResult" (
    "id" SERIAL NOT NULL,
    "hrid" VARCHAR(50) NOT NULL,
    "department_name" VARCHAR(255) NOT NULL,
    "ai_response" JSONB NOT NULL,
    "risk_score" DOUBLE PRECISION,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "AnalysisResult_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "AnalysisResult_hrid_idx" ON "AnalysisResult"("hrid");

-- CreateIndex
CREATE INDEX "AnalysisResult_department_name_idx" ON "AnalysisResult"("department_name");

-- CreateIndex
CREATE INDEX "AnalysisResult_created_at_idx" ON "AnalysisResult"("created_at");
