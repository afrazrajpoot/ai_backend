-- AlterTable
ALTER TABLE "User" ADD COLUMN     "fastApiToken" TEXT,
ADD COLUMN     "fastApiTokenExpiry" TIMESTAMP(3);

-- CreateTable
CREATE TABLE "FastAPISession" (
    "id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "email" TEXT NOT NULL,
    "token" TEXT NOT NULL,
    "expires_at" TIMESTAMP(3) NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "FastAPISession_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "FastAPISession_user_id_key" ON "FastAPISession"("user_id");
